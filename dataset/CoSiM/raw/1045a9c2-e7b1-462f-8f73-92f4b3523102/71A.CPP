#include <bits/stdc++.h>

#define ll long long

using namespace std;

int main() {
  ll i, j, x, k, n, m, c, countm = 0, countc = 0;

  ll arr[2];
  cin >> n;
  while (n--) {
    cin >> arr[0] >> arr[1];
    if (arr[0] > arr[1]) {
      countm++;
    } else if (arr[0] < arr[1]) {
      countc++;
    } else if (arr[0] == arr[1]) {
      countc++;
      countm++;
    }
  }
  if (countm > countc) {
    cout << "Mishka" << endl;
  } else if (countc > countm) {
    cout << "Chris" << endl;
  } else if (countm == countc) {
    cout << "Friendship is magic!^^" << endl;
  }
}