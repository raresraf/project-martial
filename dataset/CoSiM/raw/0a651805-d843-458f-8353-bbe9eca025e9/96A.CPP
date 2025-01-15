//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

int main() {
  string players;
  cin >> players;

  char current = players[0];
  int count = 1;

  for (int i = 1; i < players.length(); ++i) {
    if (players[i] == current) {
      count += 1;
      if (count >= 7) {
        cout << "YES";
        return 0;
      }
    } else {
      current = players[i];
      count = 1;
    }
  }

  cout << "NO";

  return 0;
}