#include <bits/stdc++.h>

#define amount_letters 26

typedef long long ll;

using namespace std;

int main() {
  string username;

  cin >> username;

  int letters[amount_letters];
  for (int i = 0; i < amount_letters; i++) {
    letters[i] = 0;
  }

  for (size_t i = 0; i < username.size(); i++) {
    int letter = (int)username[i] - 97;
    letters[letter] = 1;
  }

  int distinct_letters = 0;
  for (int i = 0; i < amount_letters; i++) {
    if (letters[i])
      distinct_letters++;
  }
  if (distinct_letters % 2 == 1)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}

// g++ -std=c++11 -O2 -Wall 236A.cpp -o out