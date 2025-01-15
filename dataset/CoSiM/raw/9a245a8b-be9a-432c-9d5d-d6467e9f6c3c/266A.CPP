#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, count = 0;
  char s[50];
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> s[i];
    if (s[i] == s[i - 1]) {
      count++;
      // cout<<s[i]<<"  "<<s[i-1]<<endl;
    }
  }
  cout << count << endl;
}
