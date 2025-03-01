/*
 * Copyright (C) 2010 Google Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1.  Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 2.  Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE AND ITS CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL APPLE OR ITS CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "third_party/blink/renderer/modules/indexeddb/idb_database.h"

#include "base/atomic_sequence_num.h"
#include "base/optional.h"
#include "third_party/blink/public/platform/modules/indexeddb/web_idb_database_callbacks.h"
#include "third_party/blink/public/platform/modules/indexeddb/web_idb_database_exception.h"
#include "third_party/blink/public/platform/modules/indexeddb/web_idb_key_path.h"
#include "third_party/blink/public/platform/modules/indexeddb/web_idb_observation.h"
#include "third_party/blink/public/platform/modules/indexeddb/web_idb_types.h"
#include "third_party/blink/renderer/bindings/core/v8/serialization/serialized_script_value.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_binding_for_modules.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_idb_observer_callback.h"
#include "third_party/blink/renderer/core/dom/events/event_queue.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/indexeddb/idb_any.h"
#include "third_party/blink/renderer/modules/indexeddb/idb_event_dispatcher.h"
#include "third_party/blink/renderer/modules/indexeddb/idb_index.h"
#include "third_party/blink/renderer/modules/indexeddb/idb_key_path.h"
#include "third_party/blink/renderer/modules/indexeddb/idb_observer.h"
#include "third_party/blink/renderer/modules/indexeddb/idb_observer_changes.h"
#include "third_party/blink/renderer/modules/indexeddb/idb_tracing.h"
#include "third_party/blink/renderer/modules/indexeddb/idb_version_change_event.h"
#include "third_party/blink/renderer/modules/indexeddb/web_idb_database_callbacks_impl.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/histogram.h"
#include "third_party/blink/renderer/platform/wtf/assertions.h"
#include "third_party/blink/renderer/platform/wtf/atomics.h"

#include <limits>
#include <memory>

using blink::WebIDBDatabase;

namespace blink {

const char IDBDatabase::kCannotObserveVersionChangeTransaction[] =
    "An observer cannot target a version change transaction.";
const char IDBDatabase::kIndexDeletedErrorMessage[] =
    "The index or its object store has been deleted.";
const char IDBDatabase::kIndexNameTakenErrorMessage[] =
    "An index with the specified name already exists.";
const char IDBDatabase::kIsKeyCursorErrorMessage[] =
    "The cursor is a key cursor.";
const char IDBDatabase::kNoKeyOrKeyRangeErrorMessage[] =
    "No key or key range specified.";
const char IDBDatabase::kNoSuchIndexErrorMessage[] =
    "The specified index was not found.";
const char IDBDatabase::kNoSuchObjectStoreErrorMessage[] =
    "The specified object store was not found.";
const char IDBDatabase::kNoValueErrorMessage[] =
    "The cursor is being iterated or has iterated past its end.";
const char IDBDatabase::kNotValidKeyErrorMessage[] =
    "The parameter is not a valid key.";
const char IDBDatabase::kNotVersionChangeTransactionErrorMessage[] =
    "The database is not running a version change transaction.";
const char IDBDatabase::kObjectStoreDeletedErrorMessage[] =
    "The object store has been deleted.";
const char IDBDatabase::kObjectStoreNameTakenErrorMessage[] =
    "An object store with the specified name already exists.";
const char IDBDatabase::kRequestNotFinishedErrorMessage[] =
    "The request has not finished.";
const char IDBDatabase::kSourceDeletedErrorMessage[] =
    "The cursor's source or effective object store has been deleted.";
const char IDBDatabase::kTransactionInactiveErrorMessage[] =
    "The transaction is not active.";
const char IDBDatabase::kTransactionFinishedErrorMessage[] =
    "The transaction has finished.";
const char IDBDatabase::kTransactionReadOnlyErrorMessage[] =
    "The transaction is read-only.";
const char IDBDatabase::kDatabaseClosedErrorMessage[] =
    "The database connection is closed.";

IDBDatabase* IDBDatabase::Create(ExecutionContext* context,
                                 std::unique_ptr<WebIDBDatabase> database,
                                 IDBDatabaseCallbacks* callbacks,
                                 v8::Isolate* isolate) {
  return new IDBDatabase(context, std::move(database), callbacks, isolate);
}

IDBDatabase::IDBDatabase(ExecutionContext* context,
                         std::unique_ptr<WebIDBDatabase> backend,
                         IDBDatabaseCallbacks* callbacks,
                         v8::Isolate* isolate)
    : ContextLifecycleObserver(context),
      backend_(std::move(backend)),
      event_queue_(EventQueue::Create(context, TaskType::kInternalIndexedDB)),
      database_callbacks_(callbacks),
      isolate_(isolate) {
  database_callbacks_->Connect(this);
}

IDBDatabase::~IDBDatabase() {
  if (!close_pending_ && backend_)
    backend_->Close();
}

void IDBDatabase::Trace(blink::Visitor* visitor) {
  visitor->Trace(version_change_transaction_);
  visitor->Trace(transactions_);
  visitor->Trace(observers_);
  visitor->Trace(event_queue_);
  visitor->Trace(database_callbacks_);
  EventTargetWithInlineData::Trace(visitor);
  ContextLifecycleObserver::Trace(visitor);
}

int64_t IDBDatabase::NextTransactionId() {
  // Starts at 1, unlike AtomicSequenceNumber.
  // Only keep a 32-bit counter to allow ports to use the other 32
  // bits of the id.
  static base::AtomicSequenceNumber current_transaction_id;
  return current_transaction_id.GetNext() + 1;
}

int32_t IDBDatabase::NextObserverId() {
  // Starts at 1, unlike AtomicSequenceNumber.
  static base::AtomicSequenceNumber current_observer_id;
  return current_observer_id.GetNext() + 1;
}

void IDBDatabase::SetMetadata(const IDBDatabaseMetadata& metadata) {
  metadata_ = metadata;
}

void IDBDatabase::SetDatabaseMetadata(const IDBDatabaseMetadata& metadata) {
  metadata_.CopyFrom(metadata);
}

void IDBDatabase::TransactionCreated(IDBTransaction* transaction) {
  DCHECK(transaction);
  DCHECK(!transactions_.Contains(transaction->Id()));
  transactions_.insert(transaction->Id(), transaction);

  if (transaction->IsVersionChange()) {
    DCHECK(!version_change_transaction_);
    version_change_transaction_ = transaction;
  }
}

void IDBDatabase::TransactionFinished(const IDBTransaction* transaction) {
  DCHECK(transaction);
  DCHECK(transactions_.Contains(transaction->Id()));
  DCHECK_EQ(transactions_.at(transaction->Id()), transaction);
  transactions_.erase(transaction->Id());

  if (transaction->IsVersionChange()) {
    DCHECK_EQ(version_change_transaction_, transaction);
    version_change_transaction_ = nullptr;
  }

  if (close_pending_ && transactions_.IsEmpty())
    CloseConnection();
}

void IDBDatabase::OnAbort(int64_t transaction_id, DOMException* error) {
  DCHECK(transactions_.Contains(transaction_id));
  transactions_.at(transaction_id)->OnAbort(error);
}

void IDBDatabase::OnComplete(int64_t transaction_id) {
  DCHECK(transactions_.Contains(transaction_id));
  transactions_.at(transaction_id)->OnComplete();
}

void IDBDatabase::OnChanges(
    const WebIDBDatabaseCallbacks::ObservationIndexMap& observation_index_map,
    WebVector<WebIDBObservation> web_observations,
    const WebIDBDatabaseCallbacks::TransactionMap& transactions) {
  HeapVector<Member<IDBObservation>> observations;
  observations.ReserveInitialCapacity(web_observations.size());
  for (WebIDBObservation& web_observation : web_observations) {
    observations.emplace_back(
        IDBObservation::Create(std::move(web_observation), isolate_));
  }

  for (const auto& map_entry : observation_index_map) {
    auto it = observers_.find(map_entry.first);
    if (it != observers_.end()) {
      IDBObserver* observer = it->value;

      IDBTransaction* transaction = nullptr;
      auto it = transactions.find(map_entry.first);
      if (it != transactions.end()) {
        const std::pair<int64_t, std::vector<int64_t>>& obs_txn = it->second;
        HashSet<String> stores;
        for (int64_t store_id : obs_txn.second) {
          stores.insert(metadata_.object_stores.at(store_id)->name);
        }

        transaction = IDBTransaction::CreateObserver(
            GetExecutionContext(), obs_txn.first, stores, this);
      }

      observer->Callback()->InvokeAndReportException(
          observer,
          IDBObserverChanges::Create(this, transaction, web_observations,
                                     observations, map_entry.second));
      if (transaction)
        transaction->SetActive(false);
    }
  }
}

DOMStringList* IDBDatabase::objectStoreNames() const {
  DOMStringList* object_store_names = DOMStringList::Create();
  for (const auto& it : metadata_.object_stores)
    object_store_names->Append(it.value->name);
  object_store_names->Sort();
  return object_store_names;
}

const String& IDBDatabase::GetObjectStoreName(int64_t object_store_id) const {
  const auto& it = metadata_.object_stores.find(object_store_id);
  DCHECK(it != metadata_.object_stores.end());
  return it->value->name;
}

int32_t IDBDatabase::AddObserver(
    IDBObserver* observer,
    int64_t transaction_id,
    bool include_transaction,
    bool no_records,
    bool values,
    const std::bitset<kWebIDBOperationTypeCount>& operation_types) {
  int32_t observer_id = NextObserverId();
  observers_.Set(observer_id, observer);
  Backend()->AddObserver(transaction_id, observer_id, include_transaction,
                         no_records, values, operation_types);
  return observer_id;
}

void IDBDatabase::RemoveObservers(const Vector<int32_t>& observer_ids) {
  observers_.RemoveAll(observer_ids);
  Backend()->RemoveObservers(observer_ids);
}

IDBObjectStore* IDBDatabase::createObjectStore(
    const String& name,
    const IDBKeyPath& key_path,
    bool auto_increment,
    ExceptionState& exception_state) {
  IDB_TRACE("IDBDatabase::createObjectStore");

  if (!version_change_transaction_) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kInvalidStateError,
        IDBDatabase::kNotVersionChangeTransactionErrorMessage);
    return nullptr;
  }
  if (!version_change_transaction_->IsActive()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kTransactionInactiveError,
        version_change_transaction_->InactiveErrorMessage());
    return nullptr;
  }

  if (!key_path.IsNull() && !key_path.IsValid()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kSyntaxError,
        "The keyPath option is not a valid key path.");
    return nullptr;
  }

  if (ContainsObjectStore(name)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kConstraintError,
        IDBDatabase::kObjectStoreNameTakenErrorMessage);
    return nullptr;
  }

  if (auto_increment && ((key_path.GetType() == IDBKeyPath::kStringType &&
                          key_path.GetString().IsEmpty()) ||
                         key_path.GetType() == IDBKeyPath::kArrayType)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kInvalidAccessError,
        "The autoIncrement option was set but the "
        "keyPath option was empty or an array.");
    return nullptr;
  }

  if (!backend_) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      IDBDatabase::kDatabaseClosedErrorMessage);
    return nullptr;
  }

  int64_t object_store_id = metadata_.max_object_store_id + 1;
  DCHECK_NE(object_store_id, IDBObjectStoreMetadata::kInvalidId);
  backend_->CreateObjectStore(version_change_transaction_->Id(),
                              object_store_id, name, key_path, auto_increment);

  scoped_refptr<IDBObjectStoreMetadata> store_metadata =
      base::AdoptRef(new IDBObjectStoreMetadata(
          name, object_store_id, key_path, auto_increment,
          WebIDBDatabase::kMinimumIndexId));
  IDBObjectStore* object_store =
      IDBObjectStore::Create(store_metadata, version_change_transaction_.Get());
  version_change_transaction_->ObjectStoreCreated(name, object_store);
  metadata_.object_stores.Set(object_store_id, std::move(store_metadata));
  ++metadata_.max_object_store_id;

  return object_store;
}

void IDBDatabase::deleteObjectStore(const String& name,
                                    ExceptionState& exception_state) {
  IDB_TRACE("IDBDatabase::deleteObjectStore");
  if (!version_change_transaction_) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kInvalidStateError,
        IDBDatabase::kNotVersionChangeTransactionErrorMessage);
    return;
  }
  if (!version_change_transaction_->IsActive()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kTransactionInactiveError,
        version_change_transaction_->InactiveErrorMessage());
    return;
  }

  int64_t object_store_id = FindObjectStoreId(name);
  if (object_store_id == IDBObjectStoreMetadata::kInvalidId) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kNotFoundError,
        "The specified object store was not found.");
    return;
  }

  if (!backend_) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      IDBDatabase::kDatabaseClosedErrorMessage);
    return;
  }

  backend_->DeleteObjectStore(version_change_transaction_->Id(),
                              object_store_id);
  version_change_transaction_->ObjectStoreDeleted(object_store_id, name);
  metadata_.object_stores.erase(object_store_id);
}

IDBTransaction* IDBDatabase::transaction(
    ScriptState* script_state,
    const StringOrStringSequence& store_names,
    const String& mode_string,
    ExceptionState& exception_state) {
  IDB_TRACE("IDBDatabase::transaction");

  HashSet<String> scope;
  if (store_names.IsString()) {
    scope.insert(store_names.GetAsString());
  } else if (store_names.IsStringSequence()) {
    for (const String& name : store_names.GetAsStringSequence())
      scope.insert(name);
  } else {
    NOTREACHED();
  }

  if (version_change_transaction_) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kInvalidStateError,
        "A version change transaction is running.");
    return nullptr;
  }

  if (close_pending_) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "The database connection is closing.");
    return nullptr;
  }

  if (!backend_) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      IDBDatabase::kDatabaseClosedErrorMessage);
    return nullptr;
  }

  if (scope.IsEmpty()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidAccessError,
                                      "The storeNames parameter was empty.");
    return nullptr;
  }

  Vector<int64_t> object_store_ids;
  for (const String& name : scope) {
    int64_t object_store_id = FindObjectStoreId(name);
    if (object_store_id == IDBObjectStoreMetadata::kInvalidId) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kNotFoundError,
          "One of the specified object stores was not found.");
      return nullptr;
    }
    object_store_ids.push_back(object_store_id);
  }

  WebIDBTransactionMode mode = IDBTransaction::StringToMode(mode_string);
  if (mode != kWebIDBTransactionModeReadOnly &&
      mode != kWebIDBTransactionModeReadWrite) {
    exception_state.ThrowTypeError(
        "The mode provided ('" + mode_string +
        "') is not one of 'readonly' or 'readwrite'.");
    return nullptr;
  }

  int64_t transaction_id = NextTransactionId();
  backend_->CreateTransaction(transaction_id, object_store_ids, mode);

  return IDBTransaction::CreateNonVersionChange(script_state, transaction_id,
                                                scope, mode, this);
}

void IDBDatabase::ForceClose() {
  for (const auto& it : transactions_)
    it.value->abort(IGNORE_EXCEPTION_FOR_TESTING);
  this->close();
  EnqueueEvent(Event::Create(EventTypeNames::close));
}

void IDBDatabase::close() {
  IDB_TRACE("IDBDatabase::close");
  if (close_pending_)
    return;

  close_pending_ = true;

  if (transactions_.IsEmpty())
    CloseConnection();
}

void IDBDatabase::CloseConnection() {
  DCHECK(close_pending_);
  DCHECK(transactions_.IsEmpty());

  if (backend_) {
    backend_->Close();
    backend_.reset();
  }

  if (database_callbacks_)
    database_callbacks_->DetachWebCallbacks();

  if (!GetExecutionContext())
    return;

  // Remove any pending versionchange events scheduled to fire on this
  // connection. They would have been scheduled by the backend when another
  // connection attempted an upgrade, but the frontend connection is being
  // closed before they could fire.
  event_queue_->CancelAllEvents();
}

void IDBDatabase::OnVersionChange(int64_t old_version, int64_t new_version) {
  IDB_TRACE("IDBDatabase::onVersionChange");
  if (!GetExecutionContext())
    return;

  if (close_pending_) {
    // If we're pending, that means there's a busy transaction. We won't
    // fire 'versionchange' but since we're not closing immediately the
    // back-end should still send out 'blocked'.
    backend_->VersionChangeIgnored();
    return;
  }

  base::Optional<unsigned long long> new_version_nullable;
  if (new_version != IDBDatabaseMetadata::kNoVersion) {
    new_version_nullable = new_version;
  }
  EnqueueEvent(IDBVersionChangeEvent::Create(
      EventTypeNames::versionchange, old_version, new_version_nullable));
}

void IDBDatabase::EnqueueEvent(Event* event) {
  DCHECK(GetExecutionContext());
  event->SetTarget(this);
  event_queue_->EnqueueEvent(FROM_HERE, event);
}

DispatchEventResult IDBDatabase::DispatchEventInternal(Event* event) {
  IDB_TRACE("IDBDatabase::dispatchEvent");
  if (!GetExecutionContext())
    return DispatchEventResult::kCanceledBeforeDispatch;
  DCHECK(event->type() == EventTypeNames::versionchange ||
         event->type() == EventTypeNames::close);

  DispatchEventResult dispatch_result =
      EventTarget::DispatchEventInternal(event);
  if (event->type() == EventTypeNames::versionchange && !close_pending_ &&
      backend_)
    backend_->VersionChangeIgnored();
  return dispatch_result;
}

int64_t IDBDatabase::FindObjectStoreId(const String& name) const {
  for (const auto& it : metadata_.object_stores) {
    if (it.value->name == name) {
      DCHECK_NE(it.key, IDBObjectStoreMetadata::kInvalidId);
      return it.key;
    }
  }
  return IDBObjectStoreMetadata::kInvalidId;
}

void IDBDatabase::RenameObjectStore(int64_t object_store_id,
                                    const String& new_name) {
  DCHECK(version_change_transaction_)
      << "Object store renamed on database without a versionchange transaction";
  DCHECK(version_change_transaction_->IsActive())
      << "Object store renamed when versionchange transaction is not active";
  DCHECK(backend_) << "Object store renamed after database connection closed";
  DCHECK(metadata_.object_stores.Contains(object_store_id));

  backend_->RenameObjectStore(version_change_transaction_->Id(),
                              object_store_id, new_name);

  IDBObjectStoreMetadata* object_store_metadata =
      metadata_.object_stores.at(object_store_id);
  version_change_transaction_->ObjectStoreRenamed(object_store_metadata->name,
                                                  new_name);
  object_store_metadata->name = new_name;
}

void IDBDatabase::RevertObjectStoreCreation(int64_t object_store_id) {
  DCHECK(version_change_transaction_) << "Object store metadata reverted on "
                                         "database without a versionchange "
                                         "transaction";
  DCHECK(!version_change_transaction_->IsActive())
      << "Object store metadata reverted when versionchange transaction is "
         "still active";
  DCHECK(metadata_.object_stores.Contains(object_store_id));
  metadata_.object_stores.erase(object_store_id);
}

void IDBDatabase::RevertObjectStoreMetadata(
    scoped_refptr<IDBObjectStoreMetadata> old_metadata) {
  DCHECK(version_change_transaction_) << "Object store metadata reverted on "
                                         "database without a versionchange "
                                         "transaction";
  DCHECK(!version_change_transaction_->IsActive())
      << "Object store metadata reverted when versionchange transaction is "
         "still active";
  DCHECK(old_metadata.get());
  metadata_.object_stores.Set(old_metadata->id, std::move(old_metadata));
}

bool IDBDatabase::HasPendingActivity() const {
  // The script wrapper must not be collected before the object is closed or
  // we can't fire a "versionchange" event to let script manually close the
  // connection.
  return !close_pending_ && GetExecutionContext() && HasEventListeners();
}

void IDBDatabase::ContextDestroyed(ExecutionContext*) {
  // Immediately close the connection to the back end. Don't attempt a
  // normal close() since that may wait on transactions which require a
  // round trip to the back-end to abort.
  if (backend_) {
    backend_->Close();
    backend_.reset();
  }

  if (database_callbacks_)
    database_callbacks_->DetachWebCallbacks();
}

const AtomicString& IDBDatabase::InterfaceName() const {
  return EventTargetNames::IDBDatabase;
}

ExecutionContext* IDBDatabase::GetExecutionContext() const {
  return ContextLifecycleObserver::GetExecutionContext();
}

STATIC_ASSERT_ENUM(kWebIDBDatabaseExceptionUnknownError,
                   DOMExceptionCode::kUnknownError);
STATIC_ASSERT_ENUM(kWebIDBDatabaseExceptionConstraintError,
                   DOMExceptionCode::kConstraintError);
STATIC_ASSERT_ENUM(kWebIDBDatabaseExceptionDataError,
                   DOMExceptionCode::kDataError);
STATIC_ASSERT_ENUM(kWebIDBDatabaseExceptionVersionError,
                   DOMExceptionCode::kVersionError);
STATIC_ASSERT_ENUM(kWebIDBDatabaseExceptionAbortError,
                   DOMExceptionCode::kAbortError);
STATIC_ASSERT_ENUM(kWebIDBDatabaseExceptionQuotaError,
                   DOMExceptionCode::kQuotaExceededError);
STATIC_ASSERT_ENUM(kWebIDBDatabaseExceptionTimeoutError,
                   DOMExceptionCode::kTimeoutError);

}  // namespace blink
