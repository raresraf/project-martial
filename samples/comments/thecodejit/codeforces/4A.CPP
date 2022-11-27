// WATERMELON

// Author :: Soumyajit Dey (TheCodejit)
#include <bits/stdc++.h>
#define ll long long
#define pb push_back
#define magic ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0)
using namespace std;
int main() {
  magic;
  int no;
  cin >> no;
  if (no == 2) {
    cout << "NO" << endl;
  } else if (no % 2 == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
