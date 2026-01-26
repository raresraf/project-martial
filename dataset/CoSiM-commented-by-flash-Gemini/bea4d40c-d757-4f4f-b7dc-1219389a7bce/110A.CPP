#include <bits/stdc++.h>
using namespace std;

int main() {
  string n;
  int count = 0;
  cin >> n;
  for (int i = 0; i < n.length(); i++) {
    n[i] = n[i] - 48;

    if ((n[i] == 4 || n[i] == 7)) {
      count++;
    }
  }
  // cout<<count<<endl;

  if (count == 4 || count == 7 || count == 47 || count == 74)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
