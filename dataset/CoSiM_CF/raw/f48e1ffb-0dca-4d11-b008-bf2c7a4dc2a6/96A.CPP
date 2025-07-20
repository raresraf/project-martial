#include <iostream>
#include <string>
using namespace std;

int main() {
  string players;
  char player;
  int counter = 0;
  cin >> players;
  char lastplayer = players.at(0);
  for (int i = 1; i < players.length(); i++) {
    player = players.at(i);
    if (player == lastplayer) {
      counter++;
      if (counter == 6) {
        break;
      }
    } else {
      counter = 0;
    }
    lastplayer = players.at(i);
  }
  if (counter < 6) {
    cout << "NO";
  } else {
    cout << "YES";
  }
  return 0;
}
