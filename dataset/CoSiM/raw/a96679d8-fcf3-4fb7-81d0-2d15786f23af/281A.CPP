// CodeForces
// Problem 281A [Word Capitalization]

#include <iostream>
using namespace std;

int main() {
  string str;
  cin >> str;

  str[0] = toupper(str[0]);
  cout << str;
  return 0;
}