#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

bool desc(char i, char j) { return i < j; }

int main() {
  string kata;
  cin >> kata;
  vector<char> coins;
  for (int i = 0; i < kata.length(); i++) {
    if (kata[i] != '+') {
      coins.push_back(kata[i]);
    }
  }

  sort(coins.begin(), coins.end(), desc);
  for (int i = 0; i < coins.size(); i++) {
    if (i != 0) {
      cout << "+";
    }
    cout << coins[i];
  }
}
