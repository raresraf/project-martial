// CodeForces
// Problem 236A [Boy or Girl]

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  string str;
  cin >> str;

  int counter = 0;

  sort(str.begin(), str.end());

  for (unsigned int i = 0; i < str.length(); i++) {
    if (str[i] != str[i + 1]) {
      counter++;
    }
  }
  if (counter % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}