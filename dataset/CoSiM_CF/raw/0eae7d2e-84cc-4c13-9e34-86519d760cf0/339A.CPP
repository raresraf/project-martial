#include <iostream>
#include <string>

using namespace std;

string sorting(string input);
void print(string sorted);

int main() {
  string input;
  string sorted = "";
  cin >> input;
  sorted = sorting(input);
  print(sorted);
  return 0;
}

string sorting(string input) {
  string ones, twos, threes;
  for (int i = 1; i < input.length(); i += 2) {
    if (input.at(i) == '+') {
      input.erase(i, 1);
    }
  }
  for (int i = 0; i < input.length(); i++) {
    if (input.at(i) == '1') {
      ones = ones + input.at(i);
    } else if (input.at(i) == '2') {
      twos = twos + input.at(i);
    } else if (input.at(i) == '3') {
      threes = threes + input.at(i);
    }
  }
  return ones + twos + threes;
}

void print(string sorted) {
  for (int i = 0; i < sorted.length(); i++) {
    cout << sorted.at(i);
    if (i != sorted.length() - 1) {
      cout << "+";
    }
  }
}