// Copyright 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stddef.h>

#include "base/guid.h"
#include "base/macros.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/utf_string_conversions.h"
#include "base/time/time.h"
#include "build/build_config.h"
#include "components/autofill/core/browser/autofill_experiments.h"
#include "components/autofill/core/browser/autofill_test_utils.h"
#include "components/autofill/core/browser/autofill_type.h"
#include "components/autofill/core/browser/credit_card.h"
#include "components/autofill/core/browser/validation.h"
#include "components/autofill/core/common/autofill_constants.h"
#include "components/autofill/core/common/form_field_data.h"
#include "components/grit/components_scaled_resources.h"
#include "components/variations/variations_params_manager.h"
#include "testing/gtest/include/gtest/gtest.h"

using base::ASCIIToUTF16;
using base::UTF8ToUTF16;

namespace autofill {

const CreditCard::RecordType LOCAL_CARD = CreditCard::LOCAL_CARD;
const CreditCard::RecordType MASKED_SERVER_CARD =
    CreditCard::MASKED_SERVER_CARD;
const CreditCard::RecordType FULL_SERVER_CARD = CreditCard::FULL_SERVER_CARD;

namespace {

// From https://www.paypalobjects.com/en_US/vhelp/paypalmanager_help/credit_card_numbers.htm
const char* const kValidNumbers[] = {
  "378282246310005",
  "3714 4963 5398 431",
  "3787-3449-3671-000",
  "5610591081018250",
  "3056 9309 0259 04",
  "3852-0000-0232-37",
  "6011111111111117",
  "6011 0009 9013 9424",
  "3530-1113-3330-0000",
  "3566002020360505",
  "5555 5555 5555 4444",
  "5105-1051-0510-5100",
  "4111111111111111",
  "4012 8888 8888 1881",
  "4222-2222-2222-2",
  "5019717010103742",
  "6331101999990016",
  "6247130048162403",
  "4532261615476013542",
  "6362970000457013",
};

const char* const kInvalidNumbers[] = {
  "4111 1111 112", /* too short */
  "41111111111111111115", /* too long */
  "4111-1111-1111-1110", /* wrong Luhn checksum */
  "3056 9309 0259 04aa", /* non-digit characters */
};

}  // namespace

TEST(CreditCardTest, GetObfuscatedStringForCardDigits) {
  const base::string16 digits = base::ASCIIToUTF16("1235");
  const base::string16 expected =
      base::string16() + base::i18n::kLeftToRightEmbeddingMark +
      kMidlineEllipsis + digits + base::i18n::kPopDirectionalFormatting;
  EXPECT_EQ(expected, internal::GetObfuscatedStringForCardDigits(digits));
}

// Tests credit card summary string generation.  This test simulates a variety
// of different possible summary strings.  Variations occur based on the
// existence of credit card number, month, and year fields.
TEST(CreditCardTest, PreviewSummaryAndNetworkAndLastFourDigitsStrings) {
  // Case 0: empty credit card.
  CreditCard credit_card0(base::GenerateGUID(), "https://www.example.com/");
  base::string16 summary0 = credit_card0.Label();
  EXPECT_EQ(base::string16(), summary0);
  base::string16 obfuscated0 = credit_card0.NetworkAndLastFourDigits();
  EXPECT_EQ(ASCIIToUTF16(std::string("Card")), obfuscated0);

  // Case 00: Empty credit card with empty strings.
  CreditCard credit_card00(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card00, "John Dillinger", "", "", "", "");
  base::string16 summary00 = credit_card00.Label();
  EXPECT_EQ(base::string16(ASCIIToUTF16("John Dillinger")), summary00);
  base::string16 obfuscated00 = credit_card00.NetworkAndLastFourDigits();
  EXPECT_EQ(ASCIIToUTF16(std::string("Card")), obfuscated00);

  // Case 1: No credit card number.
  CreditCard credit_card1(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card1, "John Dillinger", "", "01", "2010",
                          "1");
  base::string16 summary1 = credit_card1.Label();
  EXPECT_EQ(base::string16(ASCIIToUTF16("John Dillinger")), summary1);
  base::string16 obfuscated1 = credit_card1.NetworkAndLastFourDigits();
  EXPECT_EQ(ASCIIToUTF16(std::string("Card")), obfuscated1);

  // Case 2: No month.
  CreditCard credit_card2(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card2, "John Dillinger",
                          "5105 1051 0510 5100", "", "2010", "1");
  base::string16 summary2 = credit_card2.Label();
  EXPECT_EQ(UTF8ToUTF16(std::string("Mastercard  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            summary2);
  base::string16 obfuscated2 = credit_card2.NetworkAndLastFourDigits();
  EXPECT_EQ(UTF8ToUTF16(std::string("Mastercard  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            obfuscated2);

  // Case 3: No year.
  CreditCard credit_card3(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card3, "John Dillinger",
                          "5105 1051 0510 5100", "01", "", "1");
  base::string16 summary3 = credit_card3.Label();
  EXPECT_EQ(UTF8ToUTF16(std::string("Mastercard  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            summary3);
  base::string16 obfuscated3 = credit_card3.NetworkAndLastFourDigits();
  EXPECT_EQ(UTF8ToUTF16(std::string("Mastercard  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            obfuscated3);

  // Case 4: Have everything.
  CreditCard credit_card4(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card4, "John Dillinger",
                          "5105 1051 0510 5100", "01", "2010", "1");
  base::string16 summary4 = credit_card4.Label();
  EXPECT_EQ(UTF8ToUTF16(std::string("Mastercard  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100") + ", 01/2010"),
            summary4);
  base::string16 obfuscated4 = credit_card4.NetworkAndLastFourDigits();
  EXPECT_EQ(UTF8ToUTF16(std::string("Mastercard  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            obfuscated4);

  // Case 5: Very long credit card
  CreditCard credit_card5(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(
      &credit_card5, "John Dillinger",
      "0123456789 0123456789 0123456789 5105 1051 0510 5100", "01", "2010",
      "1");
  base::string16 summary5 = credit_card5.Label();
  EXPECT_EQ(UTF8ToUTF16(std::string("Card  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100") + ", 01/2010"),
            summary5);
  base::string16 obfuscated5 = credit_card5.NetworkAndLastFourDigits();
  EXPECT_EQ(UTF8ToUTF16(std::string("Card  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            obfuscated5);
}

// Tests credit card bank name and last four digits string generation.
TEST(CreditCardTest, BankNameAndLastFourDigitsStrings) {
  // Case 1: Have everything and show bank name.
  CreditCard credit_card1(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card1, "John Dillinger",
                          "5105 1051 0510 5100", "01", "2010", "1");
  credit_card1.set_bank_name("Chase");
  base::string16 obfuscated1 = credit_card1.BankNameAndLastFourDigits();
  EXPECT_FALSE(credit_card1.bank_name().empty());
  EXPECT_EQ(UTF8ToUTF16(std::string("Chase  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            obfuscated1);

  // Case 2: Have no bank name and not show bank name.
  CreditCard credit_card2(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card2, "John Dillinger",
                          "5105 1051 0510 5100", "01", "2010", "1");
  base::string16 obfuscated2 = credit_card2.BankNameAndLastFourDigits();
  EXPECT_TRUE(credit_card2.bank_name().empty());
  EXPECT_EQ(
      internal::GetObfuscatedStringForCardDigits(base::ASCIIToUTF16("5100")),
      obfuscated2);

  // Case 3: Have bank name but no last four digits, only show bank name.
  CreditCard credit_card3(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card3, "John Dillinger", "", "01", "2010",
                          "1");
  credit_card3.set_bank_name("Chase");
  base::string16 obfuscated3 = credit_card3.BankNameAndLastFourDigits();
  EXPECT_FALSE(credit_card3.bank_name().empty());
  EXPECT_EQ(UTF8ToUTF16(std::string("Chase")), obfuscated3);
}

// Tests function NetworkOrBankNameAndLastFourDigits.
TEST(CreditCardTest, NetworkOrBankNameAndLastFourDigitsStrings) {
  // Case 1: Bank name is empty -> show network name.
  CreditCard credit_card2(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card2, "John Dillinger",
                          "5105 1051 0510 5100" /* Mastercard */, "01", "2010",
                          "1");
  EXPECT_TRUE(credit_card2.bank_name().empty());
  base::string16 obfuscated2 =
      credit_card2.NetworkOrBankNameAndLastFourDigits();
  EXPECT_EQ(UTF8ToUTF16(std::string("Mastercard  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            obfuscated2);

  // Case 2: Bank name is not empty -> show bank name.
  CreditCard credit_card3(base::GenerateGUID(), "https://www.example.com/");
  test::SetCreditCardInfo(&credit_card3, "John Dillinger",
                          "5105 1051 0510 5100" /* Mastercard */, "01", "2010",
                          "1");
  credit_card3.set_bank_name("Chase");
  base::string16 obfuscated3 =
      credit_card3.NetworkOrBankNameAndLastFourDigits();
  EXPECT_FALSE(credit_card3.bank_name().empty());
  EXPECT_EQ(UTF8ToUTF16(std::string("Chase  ") +
                        test::ObfuscatedCardDigitsAsUTF8("5100")),
            obfuscated3);
}

TEST(CreditCardTest, AssignmentOperator) {
  CreditCard a(base::GenerateGUID(), test::kEmptyOrigin);
  test::SetCreditCardInfo(&a, "John Dillinger", "123456789012", "01", "2010",
                          "1");

  // Result of assignment should be logically equal to the original profile.
  CreditCard b(base::GenerateGUID(), test::kEmptyOrigin);
  b = a;
  EXPECT_EQ(a, b);

  // Assignment to self should not change the profile value.
  a = *&a;  // The *& defeats Clang's -Wself-assign warning.
  EXPECT_EQ(a, b);
}

struct SetExpirationYearFromStringTestCase {
  std::string expiration_year;
  int expected_year;
};

class SetExpirationYearFromStringTest
    : public testing::TestWithParam<SetExpirationYearFromStringTestCase> {};

TEST_P(SetExpirationYearFromStringTest, SetExpirationYearFromString) {
  auto test_case = GetParam();
  CreditCard card(base::GenerateGUID(), "some origin");
  card.SetExpirationYearFromString(ASCIIToUTF16(test_case.expiration_year));

  EXPECT_EQ(test_case.expected_year, card.expiration_year())
      << test_case.expiration_year << " " << test_case.expected_year;
}

INSTANTIATE_TEST_CASE_P(CreditCardTest,
                        SetExpirationYearFromStringTest,
                        testing::Values(
                            // Valid values.
                            SetExpirationYearFromStringTestCase{"2040", 2040},
                            SetExpirationYearFromStringTestCase{"45", 2045},
                            SetExpirationYearFromStringTestCase{"045", 2045},
                            SetExpirationYearFromStringTestCase{"9", 2009},

                            // Unrecognized year values.
                            SetExpirationYearFromStringTestCase{"052045", 0},
                            SetExpirationYearFromStringTestCase{"123", 0},
                            SetExpirationYearFromStringTestCase{"y2045", 0}));

struct SetExpirationDateFromStringTestCase {
  std::string expiration_date;
  int expected_month;
  int expected_year;
};

class SetExpirationDateFromStringTest
    : public testing::TestWithParam<SetExpirationDateFromStringTestCase> {};

TEST_P(SetExpirationDateFromStringTest, SetExpirationDateFromString) {
  auto test_case = GetParam();
  CreditCard card(base::GenerateGUID(), "some origin");
  card.SetExpirationDateFromString(ASCIIToUTF16(test_case.expiration_date));

  EXPECT_EQ(test_case.expected_month, card.expiration_month());
  EXPECT_EQ(test_case.expected_year, card.expiration_year());
}

INSTANTIATE_TEST_CASE_P(
    CreditCardTest,
    SetExpirationDateFromStringTest,
    testing::Values(
        SetExpirationDateFromStringTestCase{"10", 0, 0},       // Too small.
        SetExpirationDateFromStringTestCase{"1020451", 0, 0},  // Too long.

        // No separators.
        SetExpirationDateFromStringTestCase{"105", 0, 0},  // Too ambiguous.
        SetExpirationDateFromStringTestCase{"0545", 5, 2045},
        SetExpirationDateFromStringTestCase{"52045", 0, 0},  // Too ambiguous.
        SetExpirationDateFromStringTestCase{"052045", 5, 2045},

        // "/" separator.
        SetExpirationDateFromStringTestCase{"05/45", 5, 2045},
        SetExpirationDateFromStringTestCase{"5/2045", 5, 2045},
        SetExpirationDateFromStringTestCase{"05/2045", 5, 2045},

        // "-" separator.
        SetExpirationDateFromStringTestCase{"05-45", 5, 2045},
        SetExpirationDateFromStringTestCase{"5-2045", 5, 2045},
        SetExpirationDateFromStringTestCase{"05-2045", 5, 2045},

        // "|" separator.
        SetExpirationDateFromStringTestCase{"05|45", 5, 2045},
        SetExpirationDateFromStringTestCase{"5|2045", 5, 2045},
        SetExpirationDateFromStringTestCase{"05|2045", 5, 2045},

        // Invalid values.
        SetExpirationDateFromStringTestCase{"13/2016", 0, 2016},
        SetExpirationDateFromStringTestCase{"16/13", 0, 2013},
        SetExpirationDateFromStringTestCase{"May-2015", 0, 0},
        SetExpirationDateFromStringTestCase{"05-/2045", 0, 0},
        SetExpirationDateFromStringTestCase{"05_2045", 0, 0}));

TEST(CreditCardTest, Copy) {
  CreditCard a(base::GenerateGUID(), test::kEmptyOrigin);
  test::SetCreditCardInfo(&a, "John Dillinger", "123456789012", "01", "2010",
                          base::GenerateGUID());

  // Clone should be logically equal to the original.
  CreditCard b(a);
  EXPECT_TRUE(a == b);
}

struct IsLocalDuplicateOfServerCardTestCase {
  CreditCard::RecordType first_card_record_type;
  const char* first_card_name;
  const char* first_card_number;
  const char* first_card_exp_mo;
  const char* first_card_exp_yr;
  const char* first_billing_address_id;

  CreditCard::RecordType second_card_record_type;
  const char* second_card_name;
  const char* second_card_number;
  const char* second_card_exp_mo;
  const char* second_card_exp_yr;
  const char* second_billing_address_id;
  const char* second_card_issuer_network;

  bool is_local_duplicate;
};

class IsLocalDuplicateOfServerCardTest
    : public testing::TestWithParam<IsLocalDuplicateOfServerCardTestCase> {};

TEST_P(IsLocalDuplicateOfServerCardTest, IsLocalDuplicateOfServerCard) {
  auto test_case = GetParam();
  CreditCard a(base::GenerateGUID(), std::string());
  a.set_record_type(test_case.first_card_record_type);
  test::SetCreditCardInfo(
      &a, test_case.first_card_name, test_case.first_card_number,
      test_case.first_card_exp_mo, test_case.first_card_exp_yr,
      test_case.first_billing_address_id);

  CreditCard b(base::GenerateGUID(), std::string());
  b.set_record_type(test_case.second_card_record_type);
  test::SetCreditCardInfo(
      &b, test_case.second_card_name, test_case.second_card_number,
      test_case.second_card_exp_mo, test_case.second_card_exp_yr,
      test_case.second_billing_address_id);

  if (test_case.second_card_record_type == CreditCard::MASKED_SERVER_CARD)
    b.SetNetworkForMaskedCard(test_case.second_card_issuer_network);

  EXPECT_EQ(test_case.is_local_duplicate, a.IsLocalDuplicateOfServerCard(b))
      << " when comparing cards " << a.Label() << " and " << b.Label();
}

INSTANTIATE_TEST_CASE_P(
    CreditCardTest,
    IsLocalDuplicateOfServerCardTest,
    testing::Values(
        IsLocalDuplicateOfServerCardTestCase{LOCAL_CARD, "", "", "", "", "",
                                             LOCAL_CARD, "", "", "", "", "",
                                             nullptr, false},
        IsLocalDuplicateOfServerCardTestCase{LOCAL_CARD, "", "", "", "", "",
                                             FULL_SERVER_CARD, "", "", "", "",
                                             "", nullptr, true},
        IsLocalDuplicateOfServerCardTestCase{FULL_SERVER_CARD, "", "", "", "",
                                             "", FULL_SERVER_CARD, "", "", "",
                                             "", "", nullptr, false},
        IsLocalDuplicateOfServerCardTestCase{
            LOCAL_CARD, "John Dillinger", "423456789012", "01", "2010", "1",
            FULL_SERVER_CARD, "John Dillinger", "423456789012", "01", "2010",
            "1", nullptr, true},
        IsLocalDuplicateOfServerCardTestCase{
            LOCAL_CARD, "J Dillinger", "423456789012", "01", "2010", "1",
            FULL_SERVER_CARD, "John Dillinger", "423456789012", "01", "2010",
            "1", nullptr, false},
        IsLocalDuplicateOfServerCardTestCase{
            LOCAL_CARD, "", "423456789012", "01", "2010", "1", FULL_SERVER_CARD,
            "John Dillinger", "423456789012", "01", "2010", "1", nullptr, true},
        IsLocalDuplicateOfServerCardTestCase{
            LOCAL_CARD, "", "423456789012", "", "", "1", FULL_SERVER_CARD,
            "John Dillinger", "423456789012", "01", "2010", "1", nullptr, true},
        IsLocalDuplicateOfServerCardTestCase{
            LOCAL_CARD, "", "423456789012", "", "", "1", MASKED_SERVER_CARD,
            "John Dillinger", "9012", "01", "2010", "1", kVisaCard, true},
        IsLocalDuplicateOfServerCardTestCase{
            LOCAL_CARD, "", "423456789012", "", "", "1", MASKED_SERVER_CARD,
            "John Dillinger", "9012", "01", "2010", "1", kMasterCard, false},
        IsLocalDuplicateOfServerCardTestCase{
            LOCAL_CARD, "John Dillinger", "4234-5678-9012", "01", "2010", "1",
            FULL_SERVER_CARD, "John Dillinger", "423456789012", "01", "2010",
            "1", nullptr, true},
        IsLocalDuplicateOfServerCardTestCase{
            LOCAL_CARD, "John Dillinger", "4234-5678-9012", "01", "2010", "1",
            FULL_SERVER_CARD, "John Dillinger", "423456789012", "01", "2010",
            "2", nullptr, false}));

TEST(CreditCardTest, HasSameNumberAs) {
  CreditCard a(base::GenerateGUID(), std::string());
  CreditCard b(base::GenerateGUID(), std::string());

  // Empty cards have the same empty number.
  EXPECT_TRUE(a.HasSameNumberAs(b));
  EXPECT_TRUE(b.HasSameNumberAs(a));

  // Same number.
  a.set_record_type(CreditCard::LOCAL_CARD);
  a.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("4111111111111111"));
  a.set_record_type(CreditCard::LOCAL_CARD);
  b.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("4111111111111111"));
  EXPECT_TRUE(a.HasSameNumberAs(b));
  EXPECT_TRUE(b.HasSameNumberAs(a));

  // Local cards shouldn't match even if the last 4 are the same.
  a.set_record_type(CreditCard::LOCAL_CARD);
  a.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("4111111111111111"));
  a.set_record_type(CreditCard::LOCAL_CARD);
  b.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("4111222222221111"));
  EXPECT_FALSE(a.HasSameNumberAs(b));
  EXPECT_FALSE(b.HasSameNumberAs(a));

  // Likewise if one is an unmasked server card.
  a.set_record_type(CreditCard::FULL_SERVER_CARD);
  EXPECT_FALSE(a.HasSameNumberAs(b));
  EXPECT_FALSE(b.HasSameNumberAs(a));

  // But if one is a masked card, then they should.
  b.set_record_type(CreditCard::MASKED_SERVER_CARD);
  EXPECT_TRUE(a.HasSameNumberAs(b));
  EXPECT_TRUE(b.HasSameNumberAs(a));
}

TEST(CreditCardTest, Compare) {
  CreditCard a(base::GenerateGUID(), std::string());
  CreditCard b(base::GenerateGUID(), std::string());

  // Empty cards are the same.
  EXPECT_EQ(0, a.Compare(b));

  // GUIDs don't count.
  a.set_guid(base::GenerateGUID());
  b.set_guid(base::GenerateGUID());
  EXPECT_EQ(0, a.Compare(b));

  // Origins don't count.
  a.set_origin("apple");
  b.set_origin("banana");
  EXPECT_EQ(0, a.Compare(b));

  // Different values produce non-zero results.
  test::SetCreditCardInfo(&a, "Jimmy", nullptr, nullptr, nullptr, "");
  test::SetCreditCardInfo(&b, "Ringo", nullptr, nullptr, nullptr, "");
  EXPECT_GT(0, a.Compare(b));
  EXPECT_LT(0, b.Compare(a));
}

// Test we get the correct icon for each card type.
TEST(CreditCardTest, IconResourceId) {
  EXPECT_EQ(IDR_AUTOFILL_CC_AMEX,
            CreditCard::IconResourceId(kAmericanExpressCard));
  EXPECT_EQ(IDR_AUTOFILL_CC_DINERS,
            CreditCard::IconResourceId(kDinersCard));
  EXPECT_EQ(IDR_AUTOFILL_CC_DISCOVER,
            CreditCard::IconResourceId(kDiscoverCard));
  EXPECT_EQ(IDR_AUTOFILL_CC_ELO,
            CreditCard::IconResourceId(kEloCard));
  EXPECT_EQ(IDR_AUTOFILL_CC_JCB,
            CreditCard::IconResourceId(kJCBCard));
  EXPECT_EQ(IDR_AUTOFILL_CC_MASTERCARD,
            CreditCard::IconResourceId(kMasterCard));
  EXPECT_EQ(IDR_AUTOFILL_CC_MIR,
            CreditCard::IconResourceId(kMirCard));
  EXPECT_EQ(IDR_AUTOFILL_CC_UNIONPAY,
            CreditCard::IconResourceId(kUnionPay));
  EXPECT_EQ(IDR_AUTOFILL_CC_VISA,
            CreditCard::IconResourceId(kVisaCard));
}

TEST(CreditCardTest, UpdateFromImportedCard) {
  CreditCard original_card(base::GenerateGUID(), test::kEmptyOrigin);
  test::SetCreditCardInfo(&original_card, "John Dillinger", "123456789012",
                          "09", "2017", "1");

  CreditCard a = original_card;

  // The new card has a different name, expiration date, and origin.
  CreditCard b = a;
  b.set_guid(base::GenerateGUID());
  b.set_origin(test::kEmptyOrigin);
  b.SetRawInfo(CREDIT_CARD_NAME_FULL, ASCIIToUTF16("J. Dillinger"));
  b.SetRawInfo(CREDIT_CARD_EXP_MONTH, ASCIIToUTF16("08"));
  b.SetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR, ASCIIToUTF16("2019"));

  // |a| should be updated with the information from |b|.
  EXPECT_TRUE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ(test::kEmptyOrigin, a.origin());
  EXPECT_EQ(ASCIIToUTF16("J. Dillinger"), a.GetRawInfo(CREDIT_CARD_NAME_FULL));
  EXPECT_EQ(ASCIIToUTF16("08"), a.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(ASCIIToUTF16("2019"), a.GetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR));

  // Try again, but with no name set for |b|.
  // |a| should be updated with |b|'s expiration date and keep its original
  // name.
  a = original_card;
  b.SetRawInfo(CREDIT_CARD_NAME_FULL, base::string16());

  EXPECT_TRUE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ(test::kEmptyOrigin, a.origin());
  EXPECT_EQ(ASCIIToUTF16("John Dillinger"),
            a.GetRawInfo(CREDIT_CARD_NAME_FULL));
  EXPECT_EQ(ASCIIToUTF16("08"), a.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(ASCIIToUTF16("2019"), a.GetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR));

  // Try again, but with only the original card having a verified origin.
  // |a| should be unchanged.
  a = original_card;
  a.set_origin(kSettingsOrigin);
  b.SetRawInfo(CREDIT_CARD_NAME_FULL, ASCIIToUTF16("J. Dillinger"));

  EXPECT_TRUE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ(kSettingsOrigin, a.origin());
  EXPECT_EQ(ASCIIToUTF16("John Dillinger"),
            a.GetRawInfo(CREDIT_CARD_NAME_FULL));
  EXPECT_EQ(ASCIIToUTF16("09"), a.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(ASCIIToUTF16("2017"), a.GetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR));

  // Try again, but with using an expired verified original card.
  // |a| should not be updated because the name on the cards are not identical.
  a = original_card;
  a.set_origin("Chrome settings");
  a.SetExpirationYear(2010);

  EXPECT_TRUE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ("Chrome settings", a.origin());
  EXPECT_EQ(ASCIIToUTF16("John Dillinger"),
            a.GetRawInfo(CREDIT_CARD_NAME_FULL));
  EXPECT_EQ(ASCIIToUTF16("09"), a.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(ASCIIToUTF16("2010"), a.GetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR));

  // Try again, but with using identical name on the cards.
  // |a|'s expiration date should be updated.
  a = original_card;
  a.set_origin("Chrome settings");
  a.SetExpirationYear(2010);
  a.SetRawInfo(CREDIT_CARD_NAME_FULL, ASCIIToUTF16("J. Dillinger"));

  EXPECT_TRUE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ("Chrome settings", a.origin());
  EXPECT_EQ(ASCIIToUTF16("J. Dillinger"), a.GetRawInfo(CREDIT_CARD_NAME_FULL));
  EXPECT_EQ(ASCIIToUTF16("08"), a.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(ASCIIToUTF16("2019"), a.GetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR));

  // Try again, but with |b| being expired.
  // |a|'s expiration date should not be updated.
  a = original_card;
  a.set_origin("Chrome settings");
  a.SetExpirationYear(2010);
  a.SetRawInfo(CREDIT_CARD_NAME_FULL, ASCIIToUTF16("J. Dillinger"));
  b.SetExpirationYear(2009);

  EXPECT_TRUE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ("Chrome settings", a.origin());
  EXPECT_EQ(ASCIIToUTF16("J. Dillinger"), a.GetRawInfo(CREDIT_CARD_NAME_FULL));
  EXPECT_EQ(ASCIIToUTF16("09"), a.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(ASCIIToUTF16("2010"), a.GetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR));

  // Put back |b|'s initial expiration date.
  b.SetExpirationYear(2019);

  // Try again, but with only the new card having a verified origin.
  // |a| should be updated.
  a = original_card;
  b.set_origin(kSettingsOrigin);

  EXPECT_TRUE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ(kSettingsOrigin, a.origin());
  EXPECT_EQ(ASCIIToUTF16("J. Dillinger"), a.GetRawInfo(CREDIT_CARD_NAME_FULL));
  EXPECT_EQ(ASCIIToUTF16("08"), a.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(ASCIIToUTF16("2019"), a.GetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR));

  // Try again, with both cards having a verified origin.
  // |a| should be updated.
  a = original_card;
  a.set_origin("Chrome Autofill dialog");
  b.set_origin(kSettingsOrigin);

  EXPECT_TRUE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ(kSettingsOrigin, a.origin());
  EXPECT_EQ(ASCIIToUTF16("J. Dillinger"), a.GetRawInfo(CREDIT_CARD_NAME_FULL));
  EXPECT_EQ(ASCIIToUTF16("08"), a.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(ASCIIToUTF16("2019"), a.GetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR));

  // Try again, but with |b| having a different card number.
  // |a| should be unchanged.
  a = original_card;
  b.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("4111111111111111"));

  EXPECT_FALSE(a.UpdateFromImportedCard(b, "en-US"));
  EXPECT_EQ(original_card, a);
}

TEST(CreditCardTest, IsValidCardNumberAndExpiryDate) {
  CreditCard card;
  // Invalid because expired
  const base::Time now(base::Time::Now());
  base::Time::Exploded now_exploded;
  now.LocalExplode(&now_exploded);
  card.SetRawInfo(CREDIT_CARD_EXP_MONTH,
                  base::IntToString16(now_exploded.month));
  card.SetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR,
                  base::IntToString16(now_exploded.year - 1));
  card.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("4111111111111111"));
  EXPECT_FALSE(card.IsValid());
  EXPECT_FALSE(card.HasValidExpirationDate());
  EXPECT_TRUE(card.HasValidCardNumber());

  // Invalid because card number is not complete
  card.SetRawInfo(CREDIT_CARD_EXP_MONTH, ASCIIToUTF16("12"));
  card.SetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR, ASCIIToUTF16("2999"));
  card.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("41111"));
  EXPECT_FALSE(card.IsValid());

  for (const char* valid_number : kValidNumbers) {
    SCOPED_TRACE(valid_number);
    card.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16(valid_number));
    EXPECT_TRUE(card.IsValid());
    EXPECT_TRUE(card.HasValidCardNumber());
    EXPECT_TRUE(card.HasValidExpirationDate());
  }
  for (const char* invalid_number : kInvalidNumbers) {
    SCOPED_TRACE(invalid_number);
    card.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16(invalid_number));
    EXPECT_FALSE(card.IsValid());
    EXPECT_TRUE(card.HasValidExpirationDate());
    EXPECT_FALSE(card.HasValidCardNumber());
  }
}

// Verify that we preserve exactly what the user typed for credit card numbers.
TEST(CreditCardTest, SetRawInfoCreditCardNumber) {
  CreditCard card(base::GenerateGUID(), "https://www.example.com/");

  test::SetCreditCardInfo(&card, "Bob Dylan", "4321-5432-6543-xxxx", "07",
                          "2013", "1");
  EXPECT_EQ(ASCIIToUTF16("4321-5432-6543-xxxx"),
            card.GetRawInfo(CREDIT_CARD_NUMBER));
}

// Verify that we can handle both numeric and named months.
TEST(CreditCardTest, SetExpirationMonth) {
  CreditCard card(base::GenerateGUID(), "https://www.example.com/");

  card.SetRawInfo(CREDIT_CARD_EXP_MONTH, ASCIIToUTF16("05"));
  EXPECT_EQ(ASCIIToUTF16("05"), card.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(5, card.expiration_month());

  card.SetRawInfo(CREDIT_CARD_EXP_MONTH, ASCIIToUTF16("7"));
  EXPECT_EQ(ASCIIToUTF16("07"), card.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(7, card.expiration_month());

  // This should fail, and preserve the previous value.
  card.SetRawInfo(CREDIT_CARD_EXP_MONTH, ASCIIToUTF16("January"));
  EXPECT_EQ(ASCIIToUTF16("07"), card.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(7, card.expiration_month());

  card.SetInfo(
      AutofillType(CREDIT_CARD_EXP_MONTH), ASCIIToUTF16("January"), "en-US");
  EXPECT_EQ(ASCIIToUTF16("01"), card.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(1, card.expiration_month());

  card.SetInfo(
      AutofillType(CREDIT_CARD_EXP_MONTH), ASCIIToUTF16("Apr"), "en-US");
  EXPECT_EQ(ASCIIToUTF16("04"), card.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(4, card.expiration_month());

  card.SetInfo(AutofillType(CREDIT_CARD_EXP_MONTH),
               UTF8ToUTF16("FÉVRIER"), "fr-FR");
  EXPECT_EQ(ASCIIToUTF16("02"), card.GetRawInfo(CREDIT_CARD_EXP_MONTH));
  EXPECT_EQ(2, card.expiration_month());
}

TEST(CreditCardTest, CreditCardType) {
  CreditCard card(base::GenerateGUID(), "https://www.example.com/");

  // The card type cannot be set directly.
  card.SetRawInfo(CREDIT_CARD_TYPE, ASCIIToUTF16("Visa"));
  EXPECT_EQ(base::string16(), card.GetRawInfo(CREDIT_CARD_TYPE));

  // Setting the number should implicitly set the type.
  card.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("4111 1111 1111 1111"));
  EXPECT_EQ(ASCIIToUTF16("Visa"), card.GetRawInfo(CREDIT_CARD_TYPE));
}

TEST(CreditCardTest, CreditCardVerificationCode) {
  CreditCard card(base::GenerateGUID(), "https://www.example.com/");

  // The verification code cannot be set, as Chrome does not store this data.
  card.SetRawInfo(CREDIT_CARD_VERIFICATION_CODE, ASCIIToUTF16("999"));
  EXPECT_EQ(base::string16(), card.GetRawInfo(CREDIT_CARD_VERIFICATION_CODE));
}

struct CreditCardMatchingTypesCase {
  CreditCardMatchingTypesCase(const char* value,
                              const char* card_exp_month,
                              const char* card_exp_year,
                              CreditCard::RecordType record_type,
                              ServerFieldTypeSet expected_matched_types,
                              const char* locale = "US")
      : value(value),
        card_exp_month(card_exp_month),
        card_exp_year(card_exp_year),
        record_type(record_type),
        expected_matched_types(expected_matched_types),
        locale(locale) {}

  // The value entered by the user.
  const char* value;
  // Some values for an already saved card. Card number will be fixed to
  // 4012888888881881.
  const char* card_exp_month;
  const char* card_exp_year;
  const CreditCard::RecordType record_type;
  // The types that are expected to match.
  const ServerFieldTypeSet expected_matched_types;

  const char* locale = "US";
};

class CreditCardMatchingTypesTest
    : public testing::TestWithParam<CreditCardMatchingTypesCase> {};

TEST_P(CreditCardMatchingTypesTest, Cases) {
  auto test_case = GetParam();
  CreditCard card(base::GenerateGUID(), "https://www.example.com/");
  card.set_record_type(test_case.record_type);
  card.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("4012888888881881"));
  card.SetRawInfo(CREDIT_CARD_EXP_MONTH,
                  ASCIIToUTF16(test_case.card_exp_month));
  card.SetRawInfo(CREDIT_CARD_EXP_4_DIGIT_YEAR,
                  ASCIIToUTF16(test_case.card_exp_year));

  ServerFieldTypeSet matching_types;
  card.GetMatchingTypes(UTF8ToUTF16(test_case.value), test_case.locale,
                        &matching_types);
  EXPECT_EQ(test_case.expected_matched_types, matching_types);
}

const CreditCardMatchingTypesCase kCreditCardMatchingTypesTestCases[] = {
    // If comparing against a masked card, last four digits are checked.
    {"1881", "01", "2020", MASKED_SERVER_CARD, {CREDIT_CARD_NUMBER}},
    {"4012888888881881",
     "01",
     "2020",
     MASKED_SERVER_CARD,
     {CREDIT_CARD_NUMBER}},
    {"4111111111111111", "01", "2020", CreditCard::MASKED_SERVER_CARD,
     ServerFieldTypeSet()},
    // Same value will not match a local card or full server card since we
    // have the full number for those. However the full number will.
    {"1881", "01", "2020", LOCAL_CARD, ServerFieldTypeSet()},
    {"1881", "01", "2020", FULL_SERVER_CARD, ServerFieldTypeSet()},
    {"4012888888881881", "01", "2020", LOCAL_CARD, {CREDIT_CARD_NUMBER}},
    {"4012888888881881", "01", "2020", FULL_SERVER_CARD, {CREDIT_CARD_NUMBER}},

    // Wrong last four digits.
    {"1111", "01", "2020", MASKED_SERVER_CARD, ServerFieldTypeSet()},
    {"1111", "01", "2020", LOCAL_CARD, ServerFieldTypeSet()},
    {"1111", "01", "2020", FULL_SERVER_CARD, ServerFieldTypeSet()},
    {"4111111111111111", "01", "2020", MASKED_SERVER_CARD,
     ServerFieldTypeSet()},
    {"4111111111111111", "01", "2020", LOCAL_CARD, ServerFieldTypeSet()},
    {"4111111111111111", "01", "2020", FULL_SERVER_CARD, ServerFieldTypeSet()},

    // Matching the expiration month.
    {"01", "01", "2020", LOCAL_CARD, {CREDIT_CARD_EXP_MONTH}},
    {"1", "01", "2020", LOCAL_CARD, {CREDIT_CARD_EXP_MONTH}},
    {"jan", "01", "2020", LOCAL_CARD, {CREDIT_CARD_EXP_MONTH}, "US"},
    // Locale-specific interpretations.
    {"janv", "01", "2020", LOCAL_CARD, {CREDIT_CARD_EXP_MONTH}, "FR"},
    {"janv.", "01", "2020", LOCAL_CARD, {CREDIT_CARD_EXP_MONTH}, "FR"},
    {"janvier", "01", "2020", LOCAL_CARD, {CREDIT_CARD_EXP_MONTH}, "FR"},
    {"février", "02", "2020", LOCAL_CARD, {CREDIT_CARD_EXP_MONTH}, "FR"},
    {"mars", "01", "2020", LOCAL_CARD, ServerFieldTypeSet(), "FR"},

    // Matching the expiration year.
    {"2019", "01", "2019", LOCAL_CARD, {CREDIT_CARD_EXP_4_DIGIT_YEAR}},
    {"19", "01", "2019", LOCAL_CARD, {CREDIT_CARD_EXP_2_DIGIT_YEAR}},
    {"01/2019", "01", "2019", LOCAL_CARD, {CREDIT_CARD_EXP_DATE_4_DIGIT_YEAR}},
    {"01-2019", "01", "2019", LOCAL_CARD, {CREDIT_CARD_EXP_DATE_4_DIGIT_YEAR}},
    {"01/2020", "01", "2019", LOCAL_CARD, ServerFieldTypeSet()},
    {"20", "01", "2019", LOCAL_CARD, ServerFieldTypeSet()},
    {"2021", "01", "2019", LOCAL_CARD, ServerFieldTypeSet()},
};

INSTANTIATE_TEST_CASE_P(CreditCardTest,
                        CreditCardMatchingTypesTest,
                        testing::ValuesIn(kCreditCardMatchingTypesTestCases));

struct GetCardNetworkTestCase {
  const char* card_number;
  const char* issuer_network;
  bool is_valid;
};

// We are doing batches here because INSTANTIATE_TEST_CASE_P has a
// 50 upper limit.
class GetCardNetworkTestBatch1
    : public testing::TestWithParam<GetCardNetworkTestCase> {};

TEST_P(GetCardNetworkTestBatch1, GetCardNetwork) {
  auto test_case = GetParam();
  base::string16 card_number = ASCIIToUTF16(test_case.card_number);
  SCOPED_TRACE(card_number);
  EXPECT_EQ(test_case.issuer_network, CreditCard::GetCardNetwork(card_number));
  EXPECT_EQ(test_case.is_valid, IsValidCreditCardNumber(card_number));
}

INSTANTIATE_TEST_CASE_P(
    CreditCardTest,
    GetCardNetworkTestBatch1,
    testing::Values(
        // The relevant sample numbers from
        // http://www.paypalobjects.com/en_US/vhelp/paypalmanager_help/credit_card_numbers.htm
        GetCardNetworkTestCase{"378282246310005", kAmericanExpressCard, true},
        GetCardNetworkTestCase{"371449635398431", kAmericanExpressCard, true},
        GetCardNetworkTestCase{"378734493671000", kAmericanExpressCard, true},
        GetCardNetworkTestCase{"30569309025904", kDinersCard, true},
        GetCardNetworkTestCase{"38520000023237", kDinersCard, true},
        GetCardNetworkTestCase{"6011111111111117", kDiscoverCard, true},
        GetCardNetworkTestCase{"6011000990139424", kDiscoverCard, true},
        GetCardNetworkTestCase{"3530111333300000", kJCBCard, true},
        GetCardNetworkTestCase{"3566002020360505", kJCBCard, true},
        GetCardNetworkTestCase{"5555555555554444", kMasterCard, true},
        GetCardNetworkTestCase{"5105105105105100", kMasterCard, true},
        GetCardNetworkTestCase{"4111111111111111", kVisaCard, true},
        GetCardNetworkTestCase{"4012888888881881", kVisaCard, true},
        GetCardNetworkTestCase{"4222222222222", kVisaCard, true},
        GetCardNetworkTestCase{"4532261615476013542", kVisaCard, true},

        // The relevant sample numbers from
        // https://www.auricsystems.com/sample-credit-card-numbers/
        GetCardNetworkTestCase{"343434343434343", kAmericanExpressCard, true},
        GetCardNetworkTestCase{"371144371144376", kAmericanExpressCard, true},
        GetCardNetworkTestCase{"341134113411347", kAmericanExpressCard, true},
        GetCardNetworkTestCase{"36438936438936", kDinersCard, true},
        GetCardNetworkTestCase{"36110361103612", kDinersCard, true},
        GetCardNetworkTestCase{"36111111111111", kDinersCard, true},
        GetCardNetworkTestCase{"6011016011016011", kDiscoverCard, true},
        GetCardNetworkTestCase{"6011000990139424", kDiscoverCard, true},
        GetCardNetworkTestCase{"6011000000000004", kDiscoverCard, true},
        GetCardNetworkTestCase{"6011000995500000", kDiscoverCard, true},
        GetCardNetworkTestCase{"6500000000000002", kDiscoverCard, true},
        GetCardNetworkTestCase{"3566002020360505", kJCBCard, true},
        GetCardNetworkTestCase{"3528000000000007", kJCBCard, true},
        GetCardNetworkTestCase{"2222400061240016", kMasterCard, true},
        GetCardNetworkTestCase{"2223000048400011", kMasterCard, true},
        GetCardNetworkTestCase{"5500005555555559", kMasterCard, true},
        GetCardNetworkTestCase{"5555555555555557", kMasterCard, true},
        GetCardNetworkTestCase{"5454545454545454", kMasterCard, true},
        GetCardNetworkTestCase{"5478050000000007", kMasterCard, true},
        GetCardNetworkTestCase{"5112345112345114", kMasterCard, true},
        GetCardNetworkTestCase{"5115915115915118", kMasterCard, true},
        GetCardNetworkTestCase{"6247130048162403", kUnionPay, true},
        GetCardNetworkTestCase{"6247130048162403", kUnionPay, true},
        GetCardNetworkTestCase{"622384452162063648", kUnionPay, true},
        GetCardNetworkTestCase{"2204883716636153", kMirCard, true},
        GetCardNetworkTestCase{"2200111234567898", kMirCard, true},
        GetCardNetworkTestCase{"2200481349288130", kMirCard, true},

        // The relevant sample numbers from
        // https://www.bincodes.com/bank-creditcard-generator/ and
        // https://www.ebanx.com/business/en/developers/integrations/testing/credit-card-test-numbers
        GetCardNetworkTestCase{"5067001446391275", kEloCard, true},
        GetCardNetworkTestCase{"6362970000457013", kEloCard, true},

        // Empty string
        GetCardNetworkTestCase{"", kGenericCard, false},

        // Non-numeric
        GetCardNetworkTestCase{"garbage", kGenericCard, false},
        GetCardNetworkTestCase{"4garbage", kVisaCard, false},

        // Fails Luhn check.
        GetCardNetworkTestCase{"4111111111111112", kVisaCard, false},
        GetCardNetworkTestCase{"6247130048162413", kUnionPay, false},
        GetCardNetworkTestCase{"2204883716636154", kMirCard, false}));

class GetCardNetworkTestBatch2
    : public testing::TestWithParam<GetCardNetworkTestCase> {};

TEST_P(GetCardNetworkTestBatch2, GetCardNetwork) {
  auto test_case = GetParam();
  base::string16 card_number = ASCIIToUTF16(test_case.card_number);
  SCOPED_TRACE(card_number);
  EXPECT_EQ(test_case.issuer_network, CreditCard::GetCardNetwork(card_number));
  EXPECT_EQ(test_case.is_valid, IsValidCreditCardNumber(card_number));
}

INSTANTIATE_TEST_CASE_P(
    CreditCardTest,
    GetCardNetworkTestBatch2,
    testing::Values(
        // Invalid length.
        GetCardNetworkTestCase{"3434343434343434", kAmericanExpressCard, false},
        GetCardNetworkTestCase{"411111111111116", kVisaCard, false},
        GetCardNetworkTestCase{"220011123456783", kMirCard, false},

        // Issuer Identification Numbers (IINs) that Chrome recognizes.
        GetCardNetworkTestCase{"4", kVisaCard, false},
        GetCardNetworkTestCase{"2200", kMirCard, false},
        GetCardNetworkTestCase{"2202", kMirCard, false},
        GetCardNetworkTestCase{"2204", kMirCard, false},
        GetCardNetworkTestCase{"2221", kMasterCard, false},
        GetCardNetworkTestCase{"2720", kMasterCard, false},
        GetCardNetworkTestCase{"34", kAmericanExpressCard, false},
        GetCardNetworkTestCase{"37", kAmericanExpressCard, false},
        GetCardNetworkTestCase{"300", kDinersCard, false},
        GetCardNetworkTestCase{"301", kDinersCard, false},
        GetCardNetworkTestCase{"302", kDinersCard, false},
        GetCardNetworkTestCase{"303", kDinersCard, false},
        GetCardNetworkTestCase{"304", kDinersCard, false},
        GetCardNetworkTestCase{"305", kDinersCard, false},
        GetCardNetworkTestCase{"309", kDinersCard, false},
        GetCardNetworkTestCase{"36", kDinersCard, false},
        GetCardNetworkTestCase{"38", kDinersCard, false},
        GetCardNetworkTestCase{"39", kDinersCard, false},
        GetCardNetworkTestCase{"6011", kDiscoverCard, false},
        GetCardNetworkTestCase{"644", kDiscoverCard, false},
        GetCardNetworkTestCase{"645", kDiscoverCard, false},
        GetCardNetworkTestCase{"646", kDiscoverCard, false},
        GetCardNetworkTestCase{"647", kDiscoverCard, false},
        GetCardNetworkTestCase{"648", kDiscoverCard, false},
        GetCardNetworkTestCase{"649", kDiscoverCard, false},
        GetCardNetworkTestCase{"65", kDiscoverCard, false},
        GetCardNetworkTestCase{"5067", kEloCard, false},
        GetCardNetworkTestCase{"5090", kEloCard, false},
        GetCardNetworkTestCase{"636297", kEloCard, false},
        GetCardNetworkTestCase{"3528", kJCBCard, false},
        GetCardNetworkTestCase{"3531", kJCBCard, false},
        GetCardNetworkTestCase{"3589", kJCBCard, false},
        GetCardNetworkTestCase{"51", kMasterCard, false},
        GetCardNetworkTestCase{"52", kMasterCard, false},
        GetCardNetworkTestCase{"53", kMasterCard, false},
        GetCardNetworkTestCase{"54", kMasterCard, false},
        GetCardNetworkTestCase{"55", kMasterCard, false},
        GetCardNetworkTestCase{"62", kUnionPay, false},

        // Not enough data to determine an IIN uniquely.
        GetCardNetworkTestCase{"2", kGenericCard, false},
        GetCardNetworkTestCase{"3", kGenericCard, false},
        GetCardNetworkTestCase{"30", kGenericCard, false},
        GetCardNetworkTestCase{"35", kGenericCard, false},
        GetCardNetworkTestCase{"5", kGenericCard, false},
        GetCardNetworkTestCase{"6", kGenericCard, false},
        GetCardNetworkTestCase{"60", kGenericCard, false},
        GetCardNetworkTestCase{"601", kGenericCard, false},
        GetCardNetworkTestCase{"64", kGenericCard, false}));

class GetCardNetworkTestBatch3
    : public testing::TestWithParam<GetCardNetworkTestCase> {};

TEST_P(GetCardNetworkTestBatch3, GetCardNetwork) {
  auto test_case = GetParam();
  base::string16 card_number = ASCIIToUTF16(test_case.card_number);
  SCOPED_TRACE(card_number);
  EXPECT_EQ(test_case.issuer_network, CreditCard::GetCardNetwork(card_number));
  EXPECT_EQ(test_case.is_valid, IsValidCreditCardNumber(card_number));
}

INSTANTIATE_TEST_CASE_P(
    CreditCardTest,
    GetCardNetworkTestBatch3,
    testing::Values(
        // Unknown IINs.
        GetCardNetworkTestCase{"0", kGenericCard, false},
        GetCardNetworkTestCase{"1", kGenericCard, false},
        GetCardNetworkTestCase{"306", kGenericCard, false},
        GetCardNetworkTestCase{"307", kGenericCard, false},
        GetCardNetworkTestCase{"308", kGenericCard, false},
        GetCardNetworkTestCase{"31", kGenericCard, false},
        GetCardNetworkTestCase{"32", kGenericCard, false},
        GetCardNetworkTestCase{"33", kGenericCard, false},
        GetCardNetworkTestCase{"351", kGenericCard, false},
        GetCardNetworkTestCase{"3527", kGenericCard, false},
        GetCardNetworkTestCase{"359", kGenericCard, false},
        GetCardNetworkTestCase{"50", kGenericCard, false},
        GetCardNetworkTestCase{"56", kGenericCard, false},
        GetCardNetworkTestCase{"57", kGenericCard, false},
        GetCardNetworkTestCase{"58", kGenericCard, false},
        GetCardNetworkTestCase{"59", kGenericCard, false},
        GetCardNetworkTestCase{"600", kGenericCard, false},
        GetCardNetworkTestCase{"602", kGenericCard, false},
        GetCardNetworkTestCase{"603", kGenericCard, false},
        GetCardNetworkTestCase{"604", kGenericCard, false},
        GetCardNetworkTestCase{"605", kGenericCard, false},
        GetCardNetworkTestCase{"606", kGenericCard, false},
        GetCardNetworkTestCase{"607", kGenericCard, false},
        GetCardNetworkTestCase{"608", kGenericCard, false},
        GetCardNetworkTestCase{"609", kGenericCard, false},
        GetCardNetworkTestCase{"61", kGenericCard, false},
        GetCardNetworkTestCase{"63", kGenericCard, false},
        GetCardNetworkTestCase{"640", kGenericCard, false},
        GetCardNetworkTestCase{"641", kGenericCard, false},
        GetCardNetworkTestCase{"642", kGenericCard, false},
        GetCardNetworkTestCase{"643", kGenericCard, false},
        GetCardNetworkTestCase{"66", kGenericCard, false},
        GetCardNetworkTestCase{"67", kGenericCard, false},
        GetCardNetworkTestCase{"68", kGenericCard, false},
        GetCardNetworkTestCase{"69", kGenericCard, false},
        GetCardNetworkTestCase{"7", kGenericCard, false},
        GetCardNetworkTestCase{"8", kGenericCard, false},
        GetCardNetworkTestCase{"9", kGenericCard, false},

        // Oddball case: Unknown issuer, but valid Luhn check and plausible
        // length.
        GetCardNetworkTestCase{"7000700070007000", kGenericCard, true}));

TEST(CreditCardTest, LastFourDigits) {
  CreditCard card(base::GenerateGUID(), "https://www.example.com/");
  ASSERT_EQ(base::string16(), card.LastFourDigits());
  ASSERT_EQ(internal::GetObfuscatedStringForCardDigits(base::string16()),
            card.ObfuscatedLastFourDigits());

  test::SetCreditCardInfo(&card, "Baby Face Nelson", "5212341234123489", "01",
                          "2010", "1");
  ASSERT_EQ(base::ASCIIToUTF16("3489"), card.LastFourDigits());
  ASSERT_EQ(
      internal::GetObfuscatedStringForCardDigits(base::ASCIIToUTF16("3489")),
      card.ObfuscatedLastFourDigits());

  card.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("3489"));
  ASSERT_EQ(base::ASCIIToUTF16("3489"), card.LastFourDigits());
  ASSERT_EQ(
      internal::GetObfuscatedStringForCardDigits(base::ASCIIToUTF16("3489")),
      card.ObfuscatedLastFourDigits());

  card.SetRawInfo(CREDIT_CARD_NUMBER, ASCIIToUTF16("489"));
  ASSERT_EQ(base::ASCIIToUTF16("489"), card.LastFourDigits());
  ASSERT_EQ(
      internal::GetObfuscatedStringForCardDigits(base::ASCIIToUTF16("489")),
      card.ObfuscatedLastFourDigits());
}

// Verifies that a credit card should be updated.
struct ShouldUpdateExpirationTestCase {
  bool should_update_expiration;
  int month;
  int year;
  CreditCard::RecordType record_type;
  CreditCard::ServerStatus server_status;
};

class ShouldUpdateExpirationTest
    : public testing::TestWithParam<ShouldUpdateExpirationTestCase> {};

class TestingTimes {
 public:
  TestingTimes() {
    now_ = base::Time::Now();
    (now_ - base::TimeDelta::FromDays(365)).LocalExplode(&last_year_);
    (now_ - base::TimeDelta::FromDays(31)).LocalExplode(&last_month_);
    now_.LocalExplode(&current_);
    (now_ + base::TimeDelta::FromDays(31)).LocalExplode(&next_month_);
    (now_ + base::TimeDelta::FromDays(365)).LocalExplode(&next_year_);
  }

  base::Time now_;
  base::Time::Exploded last_year_;
  base::Time::Exploded last_month_;
  base::Time::Exploded current_;
  base::Time::Exploded next_month_;
  base::Time::Exploded next_year_;
};

TestingTimes testingTimes;

TEST_P(ShouldUpdateExpirationTest, ShouldUpdateExpiration) {
  auto test_case = GetParam();
  CreditCard card;
  card.SetExpirationMonth(test_case.month);
  card.SetExpirationYear(test_case.year);
  card.set_record_type(test_case.record_type);
  if (card.record_type() != CreditCard::LOCAL_CARD)
    card.SetServerStatus(test_case.server_status);

  EXPECT_EQ(test_case.should_update_expiration,
            card.ShouldUpdateExpiration(testingTimes.now_));
}

INSTANTIATE_TEST_CASE_P(
    CreditCardTest,
    ShouldUpdateExpirationTest,
    testing::Values(
        // Cards that expired last year should always be updated.
        ShouldUpdateExpirationTestCase{true, testingTimes.last_year_.month,
                                       testingTimes.last_year_.year,
                                       CreditCard::LOCAL_CARD},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.last_year_.month, testingTimes.last_year_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.last_year_.month, testingTimes.last_year_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.last_year_.month, testingTimes.last_year_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::EXPIRED},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.last_year_.month, testingTimes.last_year_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::EXPIRED},

        // Cards that expired last month should always be updated.
        ShouldUpdateExpirationTestCase{true, testingTimes.last_month_.month,
                                       testingTimes.last_month_.year,
                                       CreditCard::LOCAL_CARD},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.last_month_.month, testingTimes.last_month_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.last_month_.month, testingTimes.last_month_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.last_month_.month, testingTimes.last_month_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::EXPIRED},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.last_month_.month, testingTimes.last_month_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::EXPIRED},

        // Cards that expire this month should be updated only if the server
        // status is EXPIRED.
        ShouldUpdateExpirationTestCase{false, testingTimes.current_.month,
                                       testingTimes.current_.year,
                                       CreditCard::LOCAL_CARD},
        ShouldUpdateExpirationTestCase{
            false, testingTimes.current_.month, testingTimes.current_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            false, testingTimes.current_.month, testingTimes.current_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.current_.month, testingTimes.current_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::EXPIRED},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.current_.month, testingTimes.current_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::EXPIRED},

        // Cards that expire next month should be updated only if the server
        // status is EXPIRED.
        ShouldUpdateExpirationTestCase{false, testingTimes.next_month_.month,
                                       testingTimes.next_month_.year,
                                       CreditCard::LOCAL_CARD},
        ShouldUpdateExpirationTestCase{false, testingTimes.next_month_.month,
                                       testingTimes.next_month_.year,
                                       CreditCard::MASKED_SERVER_CARD,
                                       CreditCard::OK},
        ShouldUpdateExpirationTestCase{false, testingTimes.next_month_.month,
                                       testingTimes.next_month_.year,
                                       CreditCard::FULL_SERVER_CARD,
                                       CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.next_month_.month, testingTimes.next_month_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::EXPIRED},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.next_month_.month, testingTimes.next_month_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::EXPIRED},

        // Cards that expire next year should be updated only if the server
        // status is EXPIRED.
        ShouldUpdateExpirationTestCase{false, testingTimes.next_year_.month,
                                       testingTimes.next_year_.year,
                                       CreditCard::LOCAL_CARD},
        ShouldUpdateExpirationTestCase{
            false, testingTimes.next_year_.month, testingTimes.next_year_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            false, testingTimes.next_year_.month, testingTimes.next_year_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::OK},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.next_year_.month, testingTimes.next_year_.year,
            CreditCard::MASKED_SERVER_CARD, CreditCard::EXPIRED},
        ShouldUpdateExpirationTestCase{
            true, testingTimes.next_year_.month, testingTimes.next_year_.year,
            CreditCard::FULL_SERVER_CARD, CreditCard::EXPIRED}));

}  // namespace autofill
