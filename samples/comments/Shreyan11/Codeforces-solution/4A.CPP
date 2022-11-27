#include <bits/stdc++.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main() {
  int n;
  scanf("%d", &n);
  (n % 2 == 1 || n == 2)
      ? printf("NO")
      : printf("YES"); // 2 and odd numbers are the only ones who cannot be
                       // divided into 2 even numbers.
}
// Here's the problem:
// https://codeforces.com/problemset/problem/4/A