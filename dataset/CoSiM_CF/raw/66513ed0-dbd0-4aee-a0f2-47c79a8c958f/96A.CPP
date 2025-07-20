#include <bits/stdc++.h>
#define ll long long
#define vi vector<int>
#define MAX 1000
using namespace std;
void solve() {
  string a;
  cin >> a;

  int flag = 1;
  for (int i = 0; i < a.length(); i++) {
    int count = 0;
    while (a[i] == a[i + 1]) {
      count++;
    }
    if (count >= 7) {
      flag = 1;

      break;
    } else {
      flag = 0;
    }
  }
  if (flag == 0)
    cout << "no";
  else
    cout << "yes";
}

int main() {
  // int t;
  // cin>>t;
  solve();
}
