// Copyright 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "storage/browser/fileapi/file_system_operation_runner.h"

#include <stdint.h>

#include <memory>
#include <tuple>
#include <utility>

#include "base/auto_reset.h"
#include "base/bind.h"
#include "base/macros.h"
#include "base/memory/ptr_util.h"
#include "base/stl_util.h"
#include "base/threading/thread_task_runner_handle.h"
#include "net/url_request/url_request_context.h"
#include "storage/browser/blob/blob_url_request_job_factory.h"
#include "storage/browser/blob/shareable_file_reference.h"
#include "storage/browser/fileapi/file_observers.h"
#include "storage/browser/fileapi/file_stream_writer.h"
#include "storage/browser/fileapi/file_system_context.h"
#include "storage/browser/fileapi/file_writer_delegate.h"

namespace storage {

using OperationID = FileSystemOperationRunner::OperationID;

FileSystemOperationRunner::~FileSystemOperationRunner() = default;

void FileSystemOperationRunner::Shutdown() {
  // Clearing |operations_| may release our owning FileSystemContext, causing
  // |this| to be deleted, so do not touch |this| after clear()ing it.
  operations_.clear();
}

OperationID FileSystemOperationRunner::CreateFile(
    const FileSystemURL& url,
    bool exclusive,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, url);
  operation_raw->CreateFile(url, exclusive,
                            base::Bind(&FileSystemOperationRunner::DidFinish,
                                       weak_ptr_, id, callback));
  return id;
}

OperationID FileSystemOperationRunner::CreateDirectory(
    const FileSystemURL& url,
    bool exclusive,
    bool recursive,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, url);
  operation_raw->CreateDirectory(
      url, exclusive, recursive,
      base::Bind(&FileSystemOperationRunner::DidFinish, weak_ptr_, id,
                 callback));
  return id;
}

OperationID FileSystemOperationRunner::Copy(
    const FileSystemURL& src_url,
    const FileSystemURL& dest_url,
    CopyOrMoveOption option,
    ErrorBehavior error_behavior,
    const CopyProgressCallback& progress_callback,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(dest_url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, dest_url);
  PrepareForRead(id, src_url);
  operation_raw->Copy(
      src_url, dest_url, option, error_behavior,
      progress_callback.is_null()
          ? CopyProgressCallback()
          : base::Bind(&FileSystemOperationRunner::OnCopyProgress, weak_ptr_,
                       id, progress_callback),
      base::Bind(&FileSystemOperationRunner::DidFinish, weak_ptr_, id,
                 callback));
  return id;
}

OperationID FileSystemOperationRunner::Move(
    const FileSystemURL& src_url,
    const FileSystemURL& dest_url,
    CopyOrMoveOption option,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(dest_url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, dest_url);
  PrepareForWrite(id, src_url);
  operation_raw->Move(src_url, dest_url, option,
                      base::Bind(&FileSystemOperationRunner::DidFinish,
                                 weak_ptr_, id, callback));
  return id;
}

OperationID FileSystemOperationRunner::DirectoryExists(
    const FileSystemURL& url,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForRead(id, url);
  operation_raw->DirectoryExists(
      url, base::Bind(&FileSystemOperationRunner::DidFinish, weak_ptr_, id,
                      callback));
  return id;
}

OperationID FileSystemOperationRunner::FileExists(
    const FileSystemURL& url,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForRead(id, url);
  operation_raw->FileExists(
      url, base::Bind(&FileSystemOperationRunner::DidFinish, weak_ptr_, id,
                      callback));
  return id;
}

OperationID FileSystemOperationRunner::GetMetadata(
    const FileSystemURL& url,
    int fields,
    const GetMetadataCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidGetMetadata(id, callback, error, base::File::Info());
    return id;
  }
  PrepareForRead(id, url);
  operation_raw->GetMetadata(
      url, fields,
      base::Bind(&FileSystemOperationRunner::DidGetMetadata, weak_ptr_, id,
                 callback));
  return id;
}

OperationID FileSystemOperationRunner::ReadDirectory(
    const FileSystemURL& url,
    const ReadDirectoryCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidReadDirectory(id, std::move(callback), error,
                     std::vector<filesystem::mojom::DirectoryEntry>(), false);
    return id;
  }
  PrepareForRead(id, url);
  operation_raw->ReadDirectory(
      url, base::BindRepeating(&FileSystemOperationRunner::DidReadDirectory,
                               weak_ptr_, id, callback));
  return id;
}

OperationID FileSystemOperationRunner::Remove(
    const FileSystemURL& url, bool recursive,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, url);
  operation_raw->Remove(url, recursive,
                        base::Bind(&FileSystemOperationRunner::DidFinish,
                                   weak_ptr_, id, callback));
  return id;
}

OperationID FileSystemOperationRunner::Write(
    const net::URLRequestContext* url_request_context,
    const FileSystemURL& url,
    std::unique_ptr<storage::BlobDataHandle> blob,
    int64_t offset,
    const WriteCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidWrite(id, callback, error, 0, true);
    return id;
  }

  std::unique_ptr<FileStreamWriter> writer(
      file_system_context_->CreateFileStreamWriter(url, offset));
  if (!writer) {
    // Write is not supported.
    DidWrite(id, callback, base::File::FILE_ERROR_SECURITY, 0, true);
    return id;
  }

  std::unique_ptr<FileWriterDelegate> writer_delegate(new FileWriterDelegate(
      std::move(writer), url.mount_option().flush_policy()));

  std::unique_ptr<net::URLRequest> blob_request(
      storage::BlobProtocolHandler::CreateBlobRequest(
          std::move(blob), url_request_context, writer_delegate.get()));

  PrepareForWrite(id, url);
  operation_raw->Write(url, std::move(writer_delegate), std::move(blob_request),
                       base::Bind(&FileSystemOperationRunner::DidWrite,
                                  weak_ptr_, id, callback));
  return id;
}

OperationID FileSystemOperationRunner::Truncate(
    const FileSystemURL& url,
    int64_t length,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, url);
  operation_raw->Truncate(url, length,
                          base::Bind(&FileSystemOperationRunner::DidFinish,
                                     weak_ptr_, id, callback));
  return id;
}

void FileSystemOperationRunner::Cancel(
    OperationID id,
    const StatusCallback& callback) {
  if (base::ContainsKey(finished_operations_, id)) {
    DCHECK(!base::ContainsKey(stray_cancel_callbacks_, id));
    stray_cancel_callbacks_[id] = callback;
    return;
  }

  Operations::iterator found = operations_.find(id);
  if (found == operations_.end() || !found->second) {
    // There is no operation with |id|.
    callback.Run(base::File::FILE_ERROR_INVALID_OPERATION);
    return;
  }
  found->second->Cancel(callback);
}

OperationID FileSystemOperationRunner::TouchFile(
    const FileSystemURL& url,
    const base::Time& last_access_time,
    const base::Time& last_modified_time,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, url);
  operation_raw->TouchFile(url, last_access_time, last_modified_time,
                           base::Bind(&FileSystemOperationRunner::DidFinish,
                                      weak_ptr_, id, callback));
  return id;
}

OperationID FileSystemOperationRunner::OpenFile(
    const FileSystemURL& url,
    int file_flags,
    const OpenFileCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidOpenFile(id, callback, base::File(error), base::Closure());
    return id;
  }
  if (file_flags &
      (base::File::FLAG_CREATE | base::File::FLAG_OPEN_ALWAYS |
       base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_OPEN_TRUNCATED |
       base::File::FLAG_WRITE | base::File::FLAG_EXCLUSIVE_WRITE |
       base::File::FLAG_DELETE_ON_CLOSE |
       base::File::FLAG_WRITE_ATTRIBUTES)) {
    PrepareForWrite(id, url);
  } else {
    PrepareForRead(id, url);
  }
  operation_raw->OpenFile(url, file_flags,
                          base::Bind(&FileSystemOperationRunner::DidOpenFile,
                                     weak_ptr_, id, callback));
  return id;
}

OperationID FileSystemOperationRunner::CreateSnapshotFile(
    const FileSystemURL& url,
    const SnapshotFileCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidCreateSnapshot(id, callback, error, base::File::Info(), base::FilePath(),
                      NULL);
    return id;
  }
  PrepareForRead(id, url);
  operation_raw->CreateSnapshotFile(
      url, base::Bind(&FileSystemOperationRunner::DidCreateSnapshot, weak_ptr_,
                      id, callback));
  return id;
}

OperationID FileSystemOperationRunner::CopyInForeignFile(
    const base::FilePath& src_local_disk_path,
    const FileSystemURL& dest_url,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(dest_url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, dest_url);
  operation_raw->CopyInForeignFile(
      src_local_disk_path, dest_url,
      base::Bind(&FileSystemOperationRunner::DidFinish, weak_ptr_, id,
                 callback));
  return id;
}

OperationID FileSystemOperationRunner::RemoveFile(
    const FileSystemURL& url,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, url);
  operation_raw->RemoveFile(
      url, base::Bind(&FileSystemOperationRunner::DidFinish, weak_ptr_, id,
                      callback));
  return id;
}

OperationID FileSystemOperationRunner::RemoveDirectory(
    const FileSystemURL& url,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, url);
  operation_raw->RemoveDirectory(
      url, base::Bind(&FileSystemOperationRunner::DidFinish, weak_ptr_, id,
                      callback));
  return id;
}

OperationID FileSystemOperationRunner::CopyFileLocal(
    const FileSystemURL& src_url,
    const FileSystemURL& dest_url,
    CopyOrMoveOption option,
    const CopyFileProgressCallback& progress_callback,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(src_url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForRead(id, src_url);
  PrepareForWrite(id, dest_url);
  operation_raw->CopyFileLocal(src_url, dest_url, option, progress_callback,
                               base::Bind(&FileSystemOperationRunner::DidFinish,
                                          weak_ptr_, id, callback));
  return id;
}

OperationID FileSystemOperationRunner::MoveFileLocal(
    const FileSystemURL& src_url,
    const FileSystemURL& dest_url,
    CopyOrMoveOption option,
    const StatusCallback& callback) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation = base::WrapUnique(
      file_system_context_->CreateFileSystemOperation(src_url, &error));
  FileSystemOperation* operation_raw = operation.get();
  OperationID id = BeginOperation(std::move(operation));
  base::AutoReset<bool> beginning(&is_beginning_operation_, true);
  if (!operation_raw) {
    DidFinish(id, callback, error);
    return id;
  }
  PrepareForWrite(id, src_url);
  PrepareForWrite(id, dest_url);
  operation_raw->MoveFileLocal(src_url, dest_url, option,
                               base::Bind(&FileSystemOperationRunner::DidFinish,
                                          weak_ptr_, id, callback));
  return id;
}

base::File::Error FileSystemOperationRunner::SyncGetPlatformPath(
    const FileSystemURL& url,
    base::FilePath* platform_path) {
  base::File::Error error = base::File::FILE_OK;
  std::unique_ptr<FileSystemOperation> operation(
      file_system_context_->CreateFileSystemOperation(url, &error));
  if (!operation.get())
    return error;
  return operation->SyncGetPlatformPath(url, platform_path);
}

FileSystemOperationRunner::FileSystemOperationRunner(
    FileSystemContext* file_system_context)
    : file_system_context_(file_system_context), weak_factory_(this) {
  weak_ptr_ = weak_factory_.GetWeakPtr();
}

void FileSystemOperationRunner::DidFinish(const OperationID id,
                                          const StatusCallback& callback,
                                          base::File::Error rv) {
  if (is_beginning_operation_) {
    finished_operations_.insert(id);
    base::ThreadTaskRunnerHandle::Get()->PostTask(
        FROM_HERE, base::BindOnce(&FileSystemOperationRunner::DidFinish,
                                  weak_ptr_, id, callback, rv));
    return;
  }
  callback.Run(rv);
  FinishOperation(id);
}

void FileSystemOperationRunner::DidGetMetadata(
    const OperationID id,
    const GetMetadataCallback& callback,
    base::File::Error rv,
    const base::File::Info& file_info) {
  if (is_beginning_operation_) {
    finished_operations_.insert(id);
    base::ThreadTaskRunnerHandle::Get()->PostTask(
        FROM_HERE, base::BindOnce(&FileSystemOperationRunner::DidGetMetadata,
                                  weak_ptr_, id, callback, rv, file_info));
    return;
  }
  callback.Run(rv, file_info);
  FinishOperation(id);
}

void FileSystemOperationRunner::DidReadDirectory(
    const OperationID id,
    const ReadDirectoryCallback& callback,
    base::File::Error rv,
    std::vector<filesystem::mojom::DirectoryEntry> entries,
    bool has_more) {
  if (is_beginning_operation_) {
    finished_operations_.insert(id);
    base::ThreadTaskRunnerHandle::Get()->PostTask(
        FROM_HERE,
        base::BindOnce(&FileSystemOperationRunner::DidReadDirectory, weak_ptr_,
                       id, callback, rv, std::move(entries), has_more));
    return;
  }
  callback.Run(rv, std::move(entries), has_more);
  if (rv != base::File::FILE_OK || !has_more)
    FinishOperation(id);
}

void FileSystemOperationRunner::DidWrite(const OperationID id,
                                         const WriteCallback& callback,
                                         base::File::Error rv,
                                         int64_t bytes,
                                         bool complete) {
  if (is_beginning_operation_) {
    finished_operations_.insert(id);
    base::ThreadTaskRunnerHandle::Get()->PostTask(
        FROM_HERE,
        base::BindOnce(&FileSystemOperationRunner::DidWrite, weak_ptr_, id,
                       callback, rv, bytes, complete));
    return;
  }
  callback.Run(rv, bytes, complete);
  if (rv != base::File::FILE_OK || complete)
    FinishOperation(id);
}

void FileSystemOperationRunner::DidOpenFile(
    const OperationID id,
    const OpenFileCallback& callback,
    base::File file,
    base::OnceClosure on_close_callback) {
  if (is_beginning_operation_) {
    finished_operations_.insert(id);
    base::ThreadTaskRunnerHandle::Get()->PostTask(
        FROM_HERE, base::BindOnce(&FileSystemOperationRunner::DidOpenFile,
                                  weak_ptr_, id, callback, std::move(file),
                                  std::move(on_close_callback)));
    return;
  }
  callback.Run(std::move(file), std::move(on_close_callback));
  FinishOperation(id);
}

void FileSystemOperationRunner::DidCreateSnapshot(
    const OperationID id,
    const SnapshotFileCallback& callback,
    base::File::Error rv,
    const base::File::Info& file_info,
    const base::FilePath& platform_path,
    scoped_refptr<storage::ShareableFileReference> file_ref) {
  if (is_beginning_operation_) {
    finished_operations_.insert(id);
    base::ThreadTaskRunnerHandle::Get()->PostTask(
        FROM_HERE, base::BindOnce(&FileSystemOperationRunner::DidCreateSnapshot,
                                  weak_ptr_, id, callback, rv, file_info,
                                  platform_path, std::move(file_ref)));
    return;
  }
  callback.Run(rv, file_info, platform_path, std::move(file_ref));
  FinishOperation(id);
}

void FileSystemOperationRunner::OnCopyProgress(
    const OperationID id,
    const CopyProgressCallback& callback,
    FileSystemOperation::CopyProgressType type,
    const FileSystemURL& source_url,
    const FileSystemURL& dest_url,
    int64_t size) {
  if (is_beginning_operation_) {
    base::ThreadTaskRunnerHandle::Get()->PostTask(
        FROM_HERE,
        base::BindOnce(&FileSystemOperationRunner::OnCopyProgress, weak_ptr_,
                       id, callback, type, source_url, dest_url, size));
    return;
  }
  callback.Run(type, source_url, dest_url, size);
}

void FileSystemOperationRunner::PrepareForWrite(OperationID id,
                                                const FileSystemURL& url) {
  if (file_system_context_->GetUpdateObservers(url.type())) {
    file_system_context_->GetUpdateObservers(url.type())
        ->Notify(&FileUpdateObserver::OnStartUpdate, url);
  }
  write_target_urls_[id].insert(url);
}

void FileSystemOperationRunner::PrepareForRead(OperationID id,
                                               const FileSystemURL& url) {
  if (file_system_context_->GetAccessObservers(url.type())) {
    file_system_context_->GetAccessObservers(url.type())
        ->Notify(&FileAccessObserver::OnAccess, url);
  }
}

OperationID FileSystemOperationRunner::BeginOperation(
    std::unique_ptr<FileSystemOperation> operation) {
  OperationID id = next_operation_id_++;
  operations_.emplace(id, std::move(operation));
  return id;
}

void FileSystemOperationRunner::FinishOperation(OperationID id) {
  // Deleting the |operations_| entry may release the FileSystemContext which
  // owns this runner, so take a reference to keep both alive until the end of
  // this call.
  scoped_refptr<FileSystemContext> context(file_system_context_);

  OperationToURLSet::iterator found = write_target_urls_.find(id);
  if (found != write_target_urls_.end()) {
    const FileSystemURLSet& urls = found->second;
    for (const FileSystemURL& url : urls) {
      if (file_system_context_->GetUpdateObservers(url.type())) {
        file_system_context_->GetUpdateObservers(url.type())
            ->Notify(&FileUpdateObserver::OnEndUpdate, url);
      }
    }
    write_target_urls_.erase(found);
  }

  operations_.erase(id);
  finished_operations_.erase(id);

  // Dispatch stray cancel callback if exists.
  std::map<OperationID, StatusCallback>::iterator found_cancel =
      stray_cancel_callbacks_.find(id);
  if (found_cancel != stray_cancel_callbacks_.end()) {
    // This cancel has been requested after the operation has finished,
    // so report that we failed to stop it.
    found_cancel->second.Run(base::File::FILE_ERROR_INVALID_OPERATION);
    stray_cancel_callbacks_.erase(found_cancel);
  }
}

}  // namespace storage
