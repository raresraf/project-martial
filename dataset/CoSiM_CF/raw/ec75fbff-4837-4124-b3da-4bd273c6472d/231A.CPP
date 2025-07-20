#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  int a, b, c;

  int count = 0;
  while (n--) {
    cin >> a >> b >> c;
    count += (a + b + c >= 2);
  }

  cout << count << endl;
  return 0;
}