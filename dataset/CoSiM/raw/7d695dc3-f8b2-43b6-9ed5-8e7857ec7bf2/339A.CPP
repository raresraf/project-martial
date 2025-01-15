#include <algorithm>
#include <cstdio>
#include <vector>

using namespace std;

vector<int> num;

int main() {
  char x[101];
  scanf("%s", &x);
  int i;
  for (i = 0; x[i] != 0; i++) {
    if (x[i] != '+') {
      num.push_back(x[i] - '0');
    }
  }
  sort(num.begin(), num.end());
  for (i = 0; i < num.size(); i++) {
    if (i == num.size() - 1) {
      printf("%d", num[i]);
    } else {
      printf("%d+", num[i]);
    }
  }
  return 0;
}