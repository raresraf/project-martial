/*
  158A - Next Round
  CodeForces
  Esteban Arango Medina

  Solution.
    Ad-hoc, just do what they want.

*/
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
using namespace std;

int main() {
  int participants[55];
  int n, k, max;
  scanf("%d %d", &n, &k);
  for (int i = 0; i < n; ++i) {
    scanf("%d", &participants[i]);
    if (i + 1 == k)
      max = participants[i];
  }
  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (participants[i] >= max && participants[i] > 0)
      count++;
  }
  printf("%d", count);

  return 0;
}
