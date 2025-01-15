#include <bits/stdc++.h>

using namespace std;

int main() {
  string gift1, gift2;
  cin >> gift1 >> gift2;
  transform(gift1.begin(), gift1.end(), gift1.begin(), ::tolower);
  transform(gift2.begin(), gift2.end(), gift2.begin(), ::tolower);
  if (gift1 > gift2) {
    cout << "1";
  } else if (gift1 < gift2) {
    cout << "-1";
  } else if (gift1 == gift2) {
    cout << "0";
  }

  return 0;
}
