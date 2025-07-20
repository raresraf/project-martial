#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n';
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int c = 0;
  int arr[5][5] = {0};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        cout << abs(i - 2) + abs(j - 2);
        break;
      }
    }
  }

  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }