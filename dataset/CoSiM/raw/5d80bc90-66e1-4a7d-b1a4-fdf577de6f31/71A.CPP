#include <bits/stdc++.h>

// https://codeforces.com/problemset/problem/71/A

using namespace std;

int main() {
  int n, i;
  string word_input;

  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> word_input;
    if (word_input.length() <= 10) {
      cout << word_input << endl;
    } else {
      cout << word_input[0] << word_input.length() - 2
           << word_input[word_input.length() - 1] << endl;
    }
  }
}

// g++ -std=c++11 -O2 -Wall 71A.cpp -o out