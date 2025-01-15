#include <bits/stdc++.h>
using namespace std;
int main() {
  string team1ANDteam2_players;
  int counter = 1;
  cin >> team1ANDteam2_players;
  for (int i = 0; i <= team1ANDteam2_players.length(); i++) {
    if (team1ANDteam2_players[i] == team1ANDteam2_players[i + 1]) {
      counter = counter + 1;
      if (counter == 7) {
        cout << "YES";
        return 0;
      }
    } else {
      counter = 1;
    }
  }
  cout << "NO";

  return 0;
}
