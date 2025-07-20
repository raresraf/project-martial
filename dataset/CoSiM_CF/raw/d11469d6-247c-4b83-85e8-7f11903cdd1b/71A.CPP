#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  // your code goes here
  int n;
  cin >> n;

  for (int i = 0; i < n; i++) {
    string words;
    cin >> words;
    if (words.length() > 10) {
      cout << words[0] << words.length() - 2 << words[words.length() - 1]
           << endl;
    } else {
      cout << words << endl;
    }
  }

  return 0;
}
