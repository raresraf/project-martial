// Codeforces 96A - Football.cpp
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <queue>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
typedef unsigned long long llu;
#define ll long long
const double PI = acos(-1);
const double EPS = 1e-9;
const int sz = 110;
#define M 100
using namespace std;

// My Code Template Goes Here
int max(int a, int b) { return (a > b) ? a : b; }
int min(int a, int b) { return (a < b) ? a : b; }
inline int _pow(int base, int power) {
  int I, result = base;
  for (I = 0; I < power - 1; I++) {
    result = result * base;
  }
  return result;
}
// My Code Template Goes Here

// Driver function
int main() {
  char str[sz];
  scanf("%s", str);
  bool dangerous = false;
  int len = strlen(str);
  int I = 0, J, one, zero;

  while (!dangerous && I < len) {
    one = 0;
    zero = 0, J = 0;
    if (str[I] == '0') {
      J = I;
      while (str[J] == '0') {
        J++;
        zero++;
        if (zero == 7) {
          dangerous = true;
          break;
        }
      }
      I = J;
    } else if (str[I] == '1') {
      J = I;
      while (str[J] == '1') {
        J++;
        one++;
        if (one == 7) {
          dangerous = true;
          break;
        }
      }
      I = J;
    }
  }
  if (dangerous)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
