#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  string players;

  int team1 = 0, team2 = 0;

  int danger = 0;

  cin >> players;

  for (int i = 0; i < players.length(); i++) {
    if (players[i] == '0') {
      team1++;
      team2 = 0;
    } else {
      team2++;
      team1 = 0;
    }

    if (team1 == 7 || team2 == 7) {
      cout << "YES" << endl;
      danger = 1;
      break;
    }
  }

  if (!danger)
    cout << "NO" << endl;

  return 0;
}
