#include <algorithm>
#include <iostream>
#include <vector>

/*
 * Contest: Code Forces Round #111 (Task 160-A)
 * URL: http://codeforces.ru/contest/160/problem/A
 */

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(0);

  int n, sum = 0, c = 0;

  std::cin >> n;

  std::vector<int> v(n);

  for (int i = 0; i < n; ++i) {
    std::cin >> v[i];
    sum += v[i];
  }

  std::sort(v.begin(), v.end(), std::greater<int>());

  for (int i = 0; i < v.size(); ++i) {
    if ((c += v[i]) > (sum -= v[i])) {
      std::cout << (i + 1) << std::endl;
      break;
    }
  }

  return 0;
}
