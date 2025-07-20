// By archit-dev, contest: Codeforces Round #197 (Div. 2), problem: (A) Helpful
// Maths, Accepted, #, Copy
#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <string>
#define ll long long
using namespace std;

void quick_sort(int a[], int l, int u);
int partition(int a[], int l, int u);

int main() {
  string s;
  cin >> s;
  int size = s.size() / 2;
  if (s.size() % 2 != 0) {
    size += 1;
  }
  int a[size] = {0};
  int k = 0;
  for (int i = 0; i < s.size(); i += 2) {
    a[k] = s[i] - 48;
    k++;
  }
  quick_sort(a, 0, size);
  for (int i = 0; i < size; i++) {
    if (i == size - 1) {
      cout << a[i] << endl;
    } else {
      cout << a[i] << "+";
    }
  }
  return 0;
}

void quick_sort(int a[], int l, int u) {
  int j;
  if (l < u) {
    j = partition(a, l, u);
    quick_sort(a, l, j - 1);
    quick_sort(a, j + 1, u);
  }
}

int partition(int a[], int l, int u) {
  int v, i, j, temp;
  v = a[l];
  i = l;
  j = u + 1;

  do {
    do
      i++;

    while (a[i] < v && i <= u);

    do
      j--;
    while (v < a[j]);

    if (i < j) {
      temp = a[i];
      a[i] = a[j];
      a[j] = temp;
    }
  } while (i < j);

  a[l] = a[j];
  a[j] = v;

  return (j);
}
