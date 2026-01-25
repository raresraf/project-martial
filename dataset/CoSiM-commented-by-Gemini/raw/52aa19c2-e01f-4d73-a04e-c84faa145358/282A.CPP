#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  string in;
  int n;
  cin >> n;
  int res = 0;
  while (n--) {
    cin >> in;
    if (in.find("++") != in.npos)
      res++;
    else
      res--;
  }
  cout << res;
}
