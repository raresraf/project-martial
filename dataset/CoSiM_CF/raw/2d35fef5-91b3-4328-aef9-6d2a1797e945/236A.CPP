#include <algorithm>
#include <iostream>
#include <set>
#include <string>
using namespace std;

int main() {
  string word;
  cin >> word;

  set<char> word_set;

  for (char letter : word) {
    word_set.insert(letter);
  }

  if (word_set.size() % 2) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
  return 0;
}