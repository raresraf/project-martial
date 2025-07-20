#include <cstdio>
#include <iostream>
#include <list>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  int n, k;
  list<int> seq;
  scanf("%d %d", &n, &k);
  for (int i = 0; i < n; i++) {
    int x;
    scanf("%d", &x);
    seq.push_back(x);
  }

  int last_score;
  int ans = 0;
  for (int i = 0; i < k; i++) {
    if (seq.empty())
      break;
    int cur = seq.front();
    if (cur <= 0) {
      printf("%d\n", ans);
      return 0;
    }
    seq.pop_front();
    last_score = cur;
    ans++;
  }
  while (!seq.empty()) {
    int cur = seq.front();
    seq.pop_front();
    if (cur != last_score)
      break;
    ans++;
  }
  printf("%d\n", ans);
  return 0;
}
