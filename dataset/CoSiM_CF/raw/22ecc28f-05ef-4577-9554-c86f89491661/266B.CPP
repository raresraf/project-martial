
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  char ch[n + 1];
  cin >> ch;
  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (ch[i] == 'B' && ch[i + 1] == 'G') {
        swap(ch[i], ch[i + 1]);
        i++;
      }
    }
  }
  cout << ch << endl;
}
