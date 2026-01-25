#include <iostream>
using namespace std;

int main() {
  string players;
  cin >> players;
  int zeros, ones;
  string result = "NO";
  zeros = ones = 0;
  for (int i = 0; i < players.length(); i++) {
    if (players[i] == '1') {
      ones++;
      zeros = 0;
    }
    if (players[i] == '0') {
      zeros++;
      ones = 0;
    }
    if ((zeros == 7) || (ones == 7))
      result = "YES";
  }
  cout << result << "\n";
}
