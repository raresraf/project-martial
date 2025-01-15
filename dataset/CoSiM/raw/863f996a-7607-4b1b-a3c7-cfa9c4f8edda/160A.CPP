#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;
int compare(const void *a, const void *b) { return (*(int *)b - *(int *)a); }
int main() {
  int n, ar[101], sum = 0, count = 0, cheat = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> ar[i];
    sum = sum + ar[i];
  }
  qsort(ar, n, sizeof(int), compare);
  for (int j = 0; j < n; j++) {
    cheat = cheat + ar[j];
    count++;
    if (cheat > sum - cheat)
      break;
  }
  cout << count;

  return 0;
}
