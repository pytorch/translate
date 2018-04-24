#include "language_technology/neural_mt/translate/Dictionary.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "folly/experimental/TestUtil.h"

using ::folly::test::TemporaryFile;

namespace pytorch {
namespace translate {

namespace {
static TemporaryFile writeToTemp(const std::string& content) {
  TemporaryFile tempFile = TemporaryFile("test_temp");
  std::ofstream temp_stream(tempFile.path().string());
  temp_stream << content;
  return tempFile;
}
} // anonymous namespace

TEST(DictionaryTest, SpecialIdsTest) {
  TemporaryFile tempFile = writeToTemp("foo 42");
  Dictionary dict(tempFile.path().string());
  EXPECT_EQ(kPadId, dict.padId());
  EXPECT_EQ(kEosId, dict.eosId());
  EXPECT_EQ(kUnkId, dict.unkId());
}

TEST(DictionaryTest, TokenizeTest) {
  std::string line = "Ce   n'est pas \t une\nphrase!  ";
  std::vector<std::string> expected_tokens = {
      "Ce", "n'est", "pas", "une", "phrase!"};
  std::vector<std::string> tokens = Dictionary::tokenize(line);
  EXPECT_THAT(tokens, testing::ElementsAreArray(expected_tokens));
}

TEST(DictionaryTest, NumberizeAndDenumberizeTest) {
  TemporaryFile tempFile = writeToTemp(R"(foo 42
bar 30
zoohar 777
moo boo 0
)");
  Dictionary dict(tempFile.path().string());

  // Tests numberize().
  std::vector<std::string> tokens = {
      "hi",
      "foo",
      "today",
      "was",
      "moo boo",
  };
  std::vector<int> expected_ids{
      kUnkId,
      kMaxSpecialTokens,
      kUnkId,
      kUnkId,
      kMaxSpecialTokens + 3,
  };
  std::vector<int> ids = dict.numberize(tokens);
  EXPECT_THAT(ids, testing::ElementsAreArray(expected_ids));

  // Tests denumberize().
  ids = {
      kMaxSpecialTokens,
      kMaxSpecialTokens + 3,
      dict.size(),
      kMaxSpecialTokens + 2,
  };
  std::vector<std::string> expected_tokens{
      "foo",
      "moo boo",
      kUnkSymbol,
      "zoohar",
  };
  tokens = dict.denumberize(ids);
  EXPECT_THAT(tokens, testing::ElementsAreArray(expected_tokens));
}

TEST(DictionaryTest, ExceptionTest) {
  EXPECT_THROW(Dictionary dict("/non/existent/file"), std::invalid_argument);
}

} // namespace translate
} // namespace pytorch
