#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string in;
  scanf("%d", &n);
  while (n--) {
    cin >> in;
    if (in.size() > 10) {
      int len = in.size() - 2;
      in = in[0] + to_string(len) + in.back();
    }
    cout << in << endl;
  }
}
