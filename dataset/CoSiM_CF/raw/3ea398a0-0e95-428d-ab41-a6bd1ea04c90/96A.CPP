#include <bits/stdc++.h>
using namespace std;

int main() {
  string player_position;

  cin >> player_position;

  char player;
  int counter = 0;

  for (int i = 0; i < player_position.size(); i++) {

    if (i == 0) {
      player = player_position[i];
      counter++;
    } else if (player != player_position[i]) {
      player = player_position[i];
      counter = 1;
    } else
      counter++;

    if (counter == 7)
      break;
  }

  if (counter >= 7)
    cout << "YES" << endl;
  else
    cout << "NO";

  return 0;
}

/**
001001
NO

1000000001
YES

**/
