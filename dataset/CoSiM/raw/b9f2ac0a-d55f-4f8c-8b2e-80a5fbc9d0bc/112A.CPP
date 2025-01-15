#include <cctype>
#include <iostream>
#include <string>

using namespace std;
int str_compare(string &, string &);

int main() {
  string str1, str2;
  cin >> str1 >> str2;

  cout << str_compare(str1, str2) << endl;

  return 0;
}

int str_compare(string &str1, string &str2) {
  int i = 0, min_length = (str1.length() < str2.length()) ? str1.length()
                                                          : str2.length();

  int first_unmatched_char_pos = 0;

  // find out first unmatched charecter position
  while (i < min_length) {

    if (tolower(str1.at(i)) != tolower(str2.at(i))) {
      first_unmatched_char_pos = i;
      break;
    }

    i++;
  }

  int comparison_result;

  if (tolower(str1.at(first_unmatched_char_pos)) ==
      tolower(str2.at(first_unmatched_char_pos)))
    comparison_result = 0;
  else if (tolower(str1.at(first_unmatched_char_pos)) <
           tolower(str2.at(first_unmatched_char_pos)))
    comparison_result = -1;
  else
    comparison_result = 1;

  return comparison_result;
}