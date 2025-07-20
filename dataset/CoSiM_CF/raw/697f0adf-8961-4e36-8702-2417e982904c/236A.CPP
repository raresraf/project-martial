#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string input_string;
  cin >> input_string;
  int input_length(input_string.length()), iter(0),
      distinct_words(input_string.length());
  sort(input_string.begin(), input_string.end());
  for (iter = 0; iter < input_length - 1; iter++) {
    if (input_string[iter] == input_string[iter + 1]) {
      distinct_words--;
    } else {
      continue;
    }
  }
  if (distinct_words % 2 != 0) {
    cout << "IGNORE HIM!" << endl;
  } else {
    cout << "CHAT WITH HER!" << endl;
  }
  return 0;
}