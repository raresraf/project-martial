#include <bits/stdc++.h>
#define endl "\n"
#define lli long long int
#define ulli unsigned long long int

using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  int n;
  cin >> n;

  int maxcapacity = 0, passengersintram = 0;
  int a, b;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    passengersintram += b - a;

    if (passengersintram > maxcapacity) {
      maxcapacity = passengersintram;
    }
  }

  cout << maxcapacity << endl;
}