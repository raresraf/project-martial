#include <bits/stdc++.h>
using namespace std;
int main() {
  int tram = 0, in, out, n;
  int max = 0;
  scanf("%d", &n);
  getchar();
  while (n--) {
    cin >> out >> in;
    tram -= out;
    tram += in;
    if (tram > max) {
      max = tram;
    }
  }
  cout << max << endl;
}
