#include <bits/stdc++.h>

typedef long long ll;

using namespace std;

int main() {
  string players;

  cin >> players;
  int continous = 0;
  char last = '3';
  for (size_t i = 0; i < players.size(); i++) {
    if (players[i] != last) {
      last = players[i];
      continous = 0;
    }
    continous++;

    if (continous >= 7) {
      cout << "YES" << endl;
      return 0;
    }
  }
  cout << "NO" << endl;
  return 0;
}

// g++ -std=c++11 -O2 -Wall 96A.cpp -o out