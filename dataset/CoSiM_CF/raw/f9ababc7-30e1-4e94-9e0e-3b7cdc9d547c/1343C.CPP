#include <bits/stdc++.h>

using namespace std;

#define argmin(x) (min_element(begin(x), end(x)) - begin(x))
#define argmax(x) (max_element(begin(x), end(x)) - begin(x))

#define get_vector(x)                                                          \
  do {                                                                         \
    for (auto &y : x) {                                                        \
      cin >> y;                                                                \
    }                                                                          \
  } while (false)
#define put_vector(x)                                                          \
  do {                                                                         \
    for (const auto &y : x) {                                                  \
      cout << y << ' ';                                                        \
    }                                                                          \
    cout << '\n';                                                              \
  } while (false)
#define get_matrix(x)                                                          \
  do {                                                                         \
    for (auto &y : x) {                                                        \
      get_vector(y);                                                           \
    }                                                                          \
  } while (false)
#define put_matrix(x)                                                          \
  do {                                                                         \
    for (const auto &y : x) {                                                  \
      put_vector(y);                                                           \
    }                                                                          \
  } while (false)

void _main(void) {
  int n;
  cin >> n;

  vector<long long> vec;

  long long x;
  cin >> x;

  bool neg = (x < 0);
  for (int i = 1; i < n; ++i) {
    long long tmp;
    cin >> tmp;
    if (neg) {
      if (tmp < 0) {
        x = max(x, tmp);
      } else {
        vec.emplace_back(x);
        x = tmp;
        neg = false;
      }
    } else {
      if (tmp > 0) {
        x = max(x, tmp);
      } else {
        vec.emplace_back(x);
        x = tmp;
        neg = true;
      }
    }
  }
  vec.emplace_back(x);

  long long acc = 0;
  for (long long x : vec) {
    acc += x;
  }

  cout << acc << '\n';

  return;
}

int main(void) {
  cin.sync_with_stdio(false);
  cin.tie(nullptr);

  int t;
  cin >> t;
  while (t--) {
    _main();
  }

  return EXIT_SUCCESS;
}
