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

  int a;
  int k = 0;
  for (int i = 0; i < n; i++) {
    int count = 0;
    for (int j = 0; j < 3; j++) {
      cin >> a;
      if (a == 1)
        count++;
    }

    if (count >= 2) {
      k++;
    }
  }

  cout << k << endl;
}