#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, count = 0;
  char str[52];
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> str[i];
  }
  for (int i = 0; i < n - 1;
       i++) { // note : if (i<n) then str(i+1) will not found. So, i<n-1
    if (str[i] == str[i + 1]) {
      count++;
    }
  }
  cout << count << endl;
  return 0;
}
