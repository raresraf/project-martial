#include <bits/stdc++.h>
using namespace std;

int main() {
  string players;
  cin >> players;
  int count0 = 0, count1 = 0;
  for (char player0 : players) {
    if (player0 == '0') {
      count0++;
    } else {
      count0 = 0;
    }
    if (count0 == 7) {
      break;
    }
  }
  for (char player1 : players) {
    if (player1 == '1') {
      count1++;
    } else {
      count1 = 0;
    }
    if (count1 == 7) {
      break;
    }
  }
  if (count0 == 7) {
    cout << "YES" << endl;
  } else if (count1 == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
