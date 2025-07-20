#include <iostream>
#include <string>
using namespace std;
int main() {
  int no_players, kth_player, i, count, zerocount;
  int a[100000];
  cin >> no_players;
  cin >> kth_player;
  zerocount = 0;
  for (i = 0; i < no_players; i++) {
    cin >> a[i];
    if (a[i] == 0)
      zerocount++;
  }
  count = 0;
  if (zerocount != no_players) {
    for (i = 0; i < no_players; i++) {
      if (a[i] >= a[kth_player - 1] && a[i] > 0)
        count++;
    }
  }
  cout << count << endl;
  return 0;
}
