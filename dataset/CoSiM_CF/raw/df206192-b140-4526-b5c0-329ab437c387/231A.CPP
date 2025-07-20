#include <cstdio>
using namespace std;

#define rep(i, a, n) for (int i = a; i < n; i++)
#define erep(i, a, n) for (int i = a; i <= n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i++)

int main() {
  int n;
  int temp;
  int count;
  int solve = 0;
  scanf("%d", &n);
  rep(i, 0, n) {
    count = 0;
    rep(j, 0, 3) {
      scanf("%d", &temp);
      if (temp)
        count++;
    }
    if (count >= 2)
      solve++;
  }
  printf("%d\n", solve);
}
