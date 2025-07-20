#include <bits/stdc++.h>

using namespace std;
typedef std::vector<int> vi;
typedef long long ll;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  int result = 0;
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    int count = 0;
    if (a == 1)
      count++;
    if (b == 1)
      count++;
    if (c == 1)
      count++;
    if (count > 1)
      result++;
  }

  cout << result;
  return 0;
}
