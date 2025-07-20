#include <iostream>
#include <string>

using namespace std;
int minimum_stones_to_take(const string &, int);

int main() {
  int length;
  string input;
  cin >> length >> input;

  // Count the minimum number of stones to take from the table so that any two
  // neighboring stones had different colors
  cout << minimum_stones_to_take(input, length) << endl;
  return 0;
}

int minimum_stones_to_take(const string &str, int length) {
  int count = 0;

  for (int i = 0; i < length - 1; i++) {
    if (str.at(i) == str.at(i + 1))
      count++;
  }

  return count;
}