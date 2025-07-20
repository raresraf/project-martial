#include <iostream>

using namespace std;

int main() {
  int n;
  scanf(" %d", &n);
  int c = 1;
  if (n == 3) {
    puts("6");
    return 0;
  }
  int ne = 3;
  int ans = 0;
  while (c < n - 2) {
    ans += ne * (ne - 1);
    ++c;
    ++ne;
  }
  ans += ne * (ne - 1);
  cout << ans << endl;
}