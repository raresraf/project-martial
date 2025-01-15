#include <bits/stdc++.h>
using namespace std;
int main() {
  string players;
  int counter_1 = 0, counter_0 = 0;
  cin >> players;
  for (int i = 0; i <= players.size(); i++) {
    if (players[i] == '1') {
      counter_1++;
      if (counter_1 == 7) {
        break;
      }
      counter_0 = 0;
    } else if (players[i] == '0') {
      counter_0++;

      if (counter_0 == 7) {
        break;
      }
      counter_1 = 0;
    }
  }
  if (counter_0 == 7 || counter_1 == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
