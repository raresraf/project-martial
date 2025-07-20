#include <iostream>
using namespace std;
int main() {
  string player;
  int counter = 1;
  cin >> player;
  for (int i = 1; i < player.length(); i++) {
    if (player[i] == player[i - 1]) {
      counter++;
      if (counter > 6) {
        break;
      }
    } else {
      counter = 1;
    }
  }
  if (counter > 6) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
