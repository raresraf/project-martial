#include <cmath>
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int k, w;
  long long n, ans;
  scanf("%d %lld %d", &k, &n, &w);
  ans = n - k * (w * (w + 1) / 2);
  if (ans >= 0)
    printf("0\n");
  else
    printf("%lld\n", -ans);
  return 0;
}
