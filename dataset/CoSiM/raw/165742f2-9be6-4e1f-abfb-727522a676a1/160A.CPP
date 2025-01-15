#include <algorithm>
#include <cstdio>
#include <queue>

using namespace std;

int A[1000];

void merge(int beg, int mid, int end) {
  int i;
  queue<int> a, b, c;
  for (i = beg; i <= mid; ++i) {
    a.push(A[i]);
  }
  for (i = mid + 1; i <= end; ++i) {
    b.push(A[i]);
  }

  while (!a.empty() && !b.empty()) {
    if (a.front() < b.front()) {
      c.push(a.front());
      a.pop();
    } else {
      c.push(b.front());
      b.pop();
    }
  }
  while (!a.empty()) {
    c.push(a.front());
    a.pop();
  }
  while (!b.empty()) {
    c.push(b.front());
    b.pop();
  }

  i = beg;
  while (!c.empty()) {
    A[i] = c.front();
    c.pop();
    ++i;
  }
}

void mergesort(int beg, int end) {
  if (beg < end) {
    int mid = (beg + end) / 2;
    mergesort(beg, mid);
    mergesort(mid + 1, end);
    merge(beg, mid, end);
  }
}

int main() {
  int N, i, sum = 0, tsum = 0, count = 0;
  fscanf(stdin, "%d", &N);
  for (i = 0; i < N; ++i) {
    fscanf(stdin, "%d", A + i);
    sum += A[i];
  }

  mergesort(0, N - 1);

  for (i = N - 1; i >= 0; --i) {
    tsum += A[i];
    ++count;
    if (tsum > sum / 2) {
      break;
    }
  }

  fprintf(stdout, "%d", count);
  return 0;
}
