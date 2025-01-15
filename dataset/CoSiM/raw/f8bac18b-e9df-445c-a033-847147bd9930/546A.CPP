#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
int main() {
  int n;
  int start[100], end[100];
  int mark;

  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> start[i] >> end[i];
  }
  cin >> mark;

  int ch = 0;
  for (int i = 0; i < n; i++) {
    if (mark >= start[i] && mark <= end[i]) {
      break;
    }
    ch++;
  }

  cout << n - ch << endl;
}